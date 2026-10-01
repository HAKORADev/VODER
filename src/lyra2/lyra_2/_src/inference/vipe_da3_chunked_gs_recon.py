# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import argparse
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch

from lyra_2._src.inference.depth_utils import load_da3_model
from lyra_2._src.inference.vipe_da3_gs_recon import (
    DEFAULT_RECON_DA3_MODEL_PATH,
    _compute_aligned_pred_w2c,
    _ensure_da3_on_syspath,
    _import_vipe_class,
    _intrinsics_vec_to_k33,
    _interpolate_w2c,
    _load_gaussian_ply_to_gaussians,
    _pad_to_44,
    _probe_video,
    _read_video_frames_rgb,
    _save_video_mp4,
    _uniform_subsample_indices,
    _vipe_default_overrides,
)


CHUNK_ALIGNMENT_MODE = "camera_priority_joint"
CHUNK_POINT_ALIGN_WEIGHT_MODE = "inverse_depth"
CHUNK_OVERLAP_MERGE_MODE = "previous"
GS_FRUSTUM_OWNERSHIP_PRUNE = True
RENDER_CADENCE = "source"
# Total correspondence-group weights for the shared Sim(3) objective.
JOINT_CAMERA_WEIGHT = 10.0
JOINT_POINT_WEIGHT = 1.0


@dataclass(slots=True)
class ChunkSpec:
    chunk_idx: int
    source_indices: List[int]
    sampled_indices: List[int]
    keep_local_start: int

    @property
    def keep_local_indices(self) -> List[int]:
        return list(range(self.keep_local_start, len(self.source_indices)))

    @property
    def kept_source_indices(self) -> List[int]:
        return self.source_indices[self.keep_local_start :]


@dataclass(slots=True)
class PointAlignmentResult:
    r: np.ndarray
    t: np.ndarray
    s: float
    num_pairs: int
    err_mean: float
    err_p95: float
    weighted_err_mean: float = np.nan
    weight_mean: float = np.nan
    weight_p95: float = np.nan
    weight_max: float = np.nan
    weight_effective_pairs: float = np.nan
    camera_err_mean: float = np.nan
    camera_err_p95: float = np.nan


def _build_chunked_output_dir(input_video_path: str, output_dir: str | None) -> Path:
    if output_dir:
        return Path(output_dir).expanduser().resolve()

    input_video = Path(input_video_path).expanduser().resolve()
    return input_video.with_name(f"{input_video.stem}_gs_chunked")


def _target_unique_views(num_chunks: int, chunk_size: int, overlap: int) -> int:
    return int(num_chunks) * int(chunk_size) - (int(num_chunks) - 1) * int(overlap)


def _build_chunk_specs(num_chunks: int, chunk_size: int, overlap: int) -> List[ChunkSpec]:
    if num_chunks <= 0:
        raise ValueError(f"num_chunks must be positive, got {num_chunks}")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if overlap < 0:
        raise ValueError(f"chunk_overlap must be non-negative, got {overlap}")
    if num_chunks > 1 and overlap <= 0:
        raise ValueError("chunk_overlap must be > 0 when num_chunks > 1.")
    if overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({overlap}) must be smaller than chunk_size ({chunk_size})."
        )
    if num_chunks > 1 and overlap < 3:
        raise ValueError("chunk_overlap must be at least 3 so Sim(3) alignment is well-defined.")

    stride = chunk_size - overlap
    return [
        ChunkSpec(chunk_idx=chunk_idx, source_indices=[], sampled_indices=[], keep_local_start=0)
        for chunk_idx in range(num_chunks)
    ]


def _sample_exact_range(start: int, end_exclusive: int, count: int) -> List[int]:
    start = int(start)
    end_exclusive = int(end_exclusive)
    count = int(count)
    if count < 0:
        raise ValueError(f"count must be non-negative, got {count}")
    if count == 0:
        return []
    n = end_exclusive - start
    if n < count:
        raise ValueError(
            f"Cannot sample {count} unique frames from source range "
            f"[{start}, {end_exclusive}) with only {n} frames."
        )
    rel = np.floor(np.linspace(0, n - 1, num=count)).astype(np.int64)
    out = (rel + start).tolist()
    if len(set(out)) != count:
        raise RuntimeError(
            f"Internal sampling error: got duplicate frames while sampling {count} from "
            f"[{start}, {end_exclusive})."
        )
    return out


def _validate_chunk_args(num_chunks: int, chunk_size: int, overlap: int) -> None:
    _build_chunk_specs(num_chunks, chunk_size, overlap)


def _build_uniform_chunk_source_indices(
    source_total: int,
    num_chunks: int,
    chunk_size: int,
    overlap: int,
) -> tuple[np.ndarray, List[List[int]]]:
    _validate_chunk_args(num_chunks, chunk_size, overlap)
    target_unique = _target_unique_views(num_chunks, chunk_size, overlap)
    sampled_abs = np.asarray(_uniform_subsample_indices(source_total, target_unique), dtype=np.int64)
    if sampled_abs.shape[0] != target_unique:
        raise RuntimeError(f"Expected {target_unique} sampled frames, got {sampled_abs.shape[0]}.")

    stride = chunk_size - overlap
    chunks = [
        sampled_abs[chunk_idx * stride : chunk_idx * stride + chunk_size].astype(np.int64).tolist()
        for chunk_idx in range(num_chunks)
    ]
    return sampled_abs, chunks


def _build_split_chunk_source_indices(
    source_total: int,
    split_frames: Sequence[int],
    chunk_size: int,
    overlap: int,
) -> tuple[np.ndarray, List[List[int]]]:
    split_frames = [int(frame) for frame in split_frames]
    if sorted(split_frames) != split_frames or len(set(split_frames)) != len(split_frames):
        raise ValueError(f"chunk_split_frames must be strictly increasing, got {split_frames}")

    num_chunks = len(split_frames) + 1
    _validate_chunk_args(num_chunks, chunk_size, overlap)

    for split_frame in split_frames:
        if split_frame < 0 or split_frame >= source_total:
            raise ValueError(
                f"Split frame {split_frame} is outside source frame range [0, {source_total - 1}]."
            )

    chunks: List[List[int]] = []
    segment_starts = [0] + [split_frame + 1 for split_frame in split_frames]
    segment_ends = [split_frame + 1 for split_frame in split_frames] + [source_total]
    for chunk_idx in range(num_chunks):
        prev_overlap = chunks[chunk_idx - 1][-overlap:] if chunk_idx > 0 else []
        body_count = chunk_size - len(prev_overlap)
        body_start = segment_starts[chunk_idx]
        body_end = segment_ends[chunk_idx]
        body = _sample_exact_range(body_start, body_end, body_count)
        chunk_source = prev_overlap + body
        if len(chunk_source) != chunk_size:
            raise RuntimeError(
                f"Internal split sampling error: chunk {chunk_idx} has "
                f"{len(chunk_source)} frames, expected {chunk_size}."
            )
        if len(set(chunk_source)) != len(chunk_source):
            raise ValueError(
                f"Split/overlap configuration produced duplicate frames within chunk {chunk_idx}."
            )
        chunks.append(chunk_source)

    sampled_abs = np.asarray(sorted(set(frame for chunk in chunks for frame in chunk)), dtype=np.int64)
    return sampled_abs, chunks


def _build_chunk_specs_from_source_indices(
    source_indices_by_chunk: Sequence[Sequence[int]],
    sampled_abs: np.ndarray,
    overlap: int,
) -> List[ChunkSpec]:
    source_to_sampled_idx = {int(source_idx): idx for idx, source_idx in enumerate(sampled_abs.tolist())}
    specs: List[ChunkSpec] = []
    for chunk_idx, source_indices_seq in enumerate(source_indices_by_chunk):
        source_indices = [int(source_idx) for source_idx in source_indices_seq]
        sampled_indices = [source_to_sampled_idx[source_idx] for source_idx in source_indices]
        specs.append(
            ChunkSpec(
                chunk_idx=chunk_idx,
                source_indices=source_indices,
                sampled_indices=sampled_indices,
                keep_local_start=0 if chunk_idx == 0 else int(overlap),
            )
        )
    return specs


def _quat_wxyz_to_mat(quat: torch.Tensor) -> torch.Tensor:
    quat = quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    w, x, y, z = torch.unbind(quat, dim=-1)
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    return torch.stack(
        (
            ww + xx - yy - zz,
            2.0 * (xy - wz),
            2.0 * (xz + wy),
            2.0 * (xy + wz),
            ww - xx + yy - zz,
            2.0 * (yz - wx),
            2.0 * (xz - wy),
            2.0 * (yz + wx),
            ww - xx - yy + zz,
        ),
        dim=-1,
    ).reshape(quat.shape[:-1] + (3, 3))


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.clamp(x, min=0.0))


def _mat_to_quat_wxyz(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"Expected rotation matrices (...,3,3), got {tuple(matrix.shape)}")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)),
        dim=-1,
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    quat_by_wxyz = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m02 + m20, m12 + m21, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )
    floor = torch.tensor(0.1, dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_wxyz / (2.0 * q_abs[..., None].max(floor))
    out = quat_candidates[
        torch.nn.functional.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5,
        :,
    ].reshape(batch_dim + (4,))
    return torch.where(out[..., :1] < 0, -out, out)


def _apply_sim3_to_w2c(w2c: np.ndarray, r: np.ndarray, t: np.ndarray, s: float) -> np.ndarray:
    _ensure_da3_on_syspath()
    from depth_anything_3.utils.geometry import affine_inverse_np  # type: ignore

    w2c_44 = _pad_to_44(w2c)
    c2w = affine_inverse_np(w2c_44)
    c2w_aligned = c2w.copy()
    c2w_aligned[:, :3, :3] = np.einsum("ij,njk->nik", r, c2w[:, :3, :3])
    c2w_aligned[:, :3, 3] = np.einsum("ij,nj->ni", r, float(s) * c2w[:, :3, 3]) + t[None]
    c2w_aligned[:, 3, 3] = 1.0
    return affine_inverse_np(c2w_aligned).astype(np.float32)


def _apply_sim3_to_gaussians(gaussians, r: np.ndarray, t: np.ndarray, s: float):
    _ensure_da3_on_syspath()
    from depth_anything_3.specs import Gaussians  # type: ignore

    means = gaussians.means
    device = means.device
    dtype = means.dtype
    r_t = torch.as_tensor(r, device=device, dtype=dtype)
    t_t = torch.as_tensor(t, device=device, dtype=dtype)
    s_t = torch.as_tensor(float(s), device=device, dtype=dtype)

    means = torch.einsum("ij,bnj->bni", r_t, means * s_t) + t_t.view(1, 1, 3)
    scales = gaussians.scales * s_t

    rot_mats = _quat_wxyz_to_mat(gaussians.rotations)
    rot_mats = r_t.view(1, 1, 3, 3) @ rot_mats
    rotations = _mat_to_quat_wxyz(rot_mats).to(dtype=gaussians.rotations.dtype)

    return Gaussians(
        means=means,
        scales=scales,
        rotations=rotations,
        harmonics=gaussians.harmonics,
        opacities=gaussians.opacities,
    )


def _select_gaussian_views(gaussians, view_indices: Sequence[int], num_views: int):
    _ensure_da3_on_syspath()
    from depth_anything_3.specs import Gaussians  # type: ignore

    if not view_indices:
        raise ValueError("Cannot select zero gaussian views.")

    index = torch.as_tensor(list(view_indices), device=gaussians.means.device, dtype=torch.long)

    def _select(attr: torch.Tensor) -> torch.Tensor:
        if attr.shape[1] % num_views != 0:
            raise ValueError(
                f"Gaussian attr has {attr.shape[1]} entries, not divisible by {num_views} views."
            )
        per_view = attr.shape[1] // num_views
        shaped = attr.reshape(attr.shape[0], num_views, per_view, *attr.shape[2:])
        selected = shaped.index_select(1, index)
        return selected.reshape(attr.shape[0], len(view_indices) * per_view, *attr.shape[2:])

    return Gaussians(
        means=_select(gaussians.means),
        scales=_select(gaussians.scales),
        rotations=_select(gaussians.rotations),
        harmonics=_select(gaussians.harmonics),
        opacities=_select(gaussians.opacities),
    )


def _chunk_gs_keep_indices(
    chunk_idx: int,
    num_views: int,
    overlap: int,
) -> List[int]:
    start = int(overlap) if chunk_idx > 0 else 0
    return list(range(start, num_views))


def _alignment_point_indices(n_per_view: int, max_points_per_view: int) -> np.ndarray:
    n_per_view = int(n_per_view)
    max_points_per_view = int(max_points_per_view)
    if max_points_per_view <= 0 or max_points_per_view >= n_per_view:
        return np.arange(n_per_view, dtype=np.int64)
    return np.linspace(0, n_per_view - 1, num=max_points_per_view, dtype=np.int64)


def _extract_gaussian_alignment_points(
    gaussians,
    view_indices: Sequence[int],
    num_views: int,
    max_points_per_view: int,
) -> List[np.ndarray]:
    means = gaussians.means[0].detach().cpu()
    n_total = int(means.shape[0])
    if n_total % int(num_views) != 0:
        raise ValueError(f"Gaussian count {n_total} is not divisible by {num_views} views.")
    per_view = n_total // int(num_views)
    point_idx = torch.as_tensor(
        _alignment_point_indices(per_view, max_points_per_view),
        dtype=torch.long,
    )
    means_by_view = means.reshape(int(num_views), per_view, 3)
    out = []
    for view_idx in view_indices:
        pts = means_by_view[int(view_idx)].index_select(0, point_idx).numpy().astype(np.float32)
        valid = np.isfinite(pts).all(axis=1)
        out.append(pts[valid])
    return out


def _fit_sim3_points_umeyama(
    source_points: np.ndarray,
    target_points: np.ndarray,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    source = np.asarray(source_points, dtype=np.float64).reshape(-1, 3)
    target = np.asarray(target_points, dtype=np.float64).reshape(-1, 3)
    if source.shape != target.shape:
        raise ValueError(f"Point shape mismatch: {source.shape} vs {target.shape}.")
    if source.shape[0] < 3:
        raise ValueError(f"Need at least 3 point pairs, got {source.shape[0]}.")
    weight = _positive_normalized_weights(weights, source.shape[0]).astype(np.float64)
    weight = weight / max(float(weight.sum()), 1e-12)

    mu_source = np.sum(source * weight[:, None], axis=0)
    mu_target = np.sum(target * weight[:, None], axis=0)
    source_centered = source - mu_source
    target_centered = target - mu_target
    var_source = float(np.sum(weight * np.sum(source_centered * source_centered, axis=1)))
    if var_source < 1e-12:
        raise ValueError("Source points are degenerate for Sim3 alignment.")

    covariance = source_centered.T @ (target_centered * weight[:, None])
    u, singular_values, vt = np.linalg.svd(covariance)
    d = np.ones(3, dtype=np.float64)
    if np.linalg.det(vt.T @ u.T) < 0:
        d[-1] = -1.0
    r = vt.T @ np.diag(d) @ u.T
    scale = float(np.sum(singular_values * d) / var_source)
    t = mu_target - scale * (r @ mu_source)
    return r.astype(np.float32), t.astype(np.float32), scale


def _apply_sim3_to_points(points: np.ndarray, r: np.ndarray, t: np.ndarray, s: float) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    return (np.einsum("ij,nj->ni", r, pts * float(s)) + t[None]).astype(np.float32)


def _positive_normalized_weights(weights: np.ndarray | None, n: int) -> np.ndarray:
    if weights is None:
        return np.ones(int(n), dtype=np.float32)
    out = np.asarray(weights, dtype=np.float32).reshape(-1)
    if out.shape[0] != int(n):
        raise ValueError(f"Weight count mismatch: {out.shape[0]} vs {n}.")
    out = np.where(np.isfinite(out) & (out > 0.0), out, 0.0).astype(np.float32)
    if float(out.sum()) <= 0.0:
        return np.ones(int(n), dtype=np.float32)
    return (out / np.mean(out[out > 0.0])).astype(np.float32)


def _weighted_mean(values: np.ndarray, weights: np.ndarray | None = None) -> float:
    val = np.asarray(values, dtype=np.float64).reshape(-1)
    if val.size == 0:
        return float("nan")
    if weights is None:
        return float(val.mean())
    w = _positive_normalized_weights(weights, val.size).astype(np.float64)
    return float(np.sum(val * w) / max(float(np.sum(w)), 1e-12))


def _weighted_effective_count(weights: np.ndarray | None, n: int) -> float:
    if weights is None:
        return float(n)
    w = _positive_normalized_weights(weights, n).astype(np.float64)
    denom = float(np.sum(w * w))
    if denom <= 1e-12:
        return 0.0
    return float(np.sum(w) ** 2 / denom)


def _camera_centers_from_w2c(w2c: np.ndarray) -> np.ndarray:
    c2w = np.linalg.inv(_pad_to_44(np.asarray(w2c, dtype=np.float32)))
    return c2w[:, :3, 3].astype(np.float32)


def _point_depths_in_camera(points: np.ndarray, w2c: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    w2c_44 = _pad_to_44(np.asarray(w2c, dtype=np.float32))
    if w2c_44.ndim == 3:
        w2c_44 = w2c_44[0]
    cam = np.einsum("ij,nj->ni", w2c_44[:3, :3], pts) + w2c_44[:3, 3][None]
    return cam[:, 2].astype(np.float32)


def _inverse_depth_weights(
    depths: np.ndarray,
    *,
    power: float,
    clip_percentile: float,
    min_depth: float,
) -> np.ndarray:
    z = np.asarray(depths, dtype=np.float32).reshape(-1)
    valid = np.isfinite(z) & (z > float(min_depth))
    weights = np.zeros_like(z, dtype=np.float32)
    if not np.any(valid):
        return np.ones_like(z, dtype=np.float32)

    inv = 1.0 / np.clip(z[valid], float(min_depth), None)
    power = float(power)
    if power != 1.0:
        inv = inv**power
    clip_percentile = float(clip_percentile)
    if 0.0 < clip_percentile < 1.0 and inv.size > 1:
        inv = np.minimum(inv, float(np.quantile(inv, clip_percentile)))
    weights[valid] = inv.astype(np.float32)
    return _positive_normalized_weights(weights, len(weights))


def _collect_overlap_point_pairs(
    gaussians,
    overlap_source: Sequence[int],
    alignment_points_by_source_idx: dict[int, np.ndarray],
    *,
    chunk_overlap: int,
    num_views: int,
    max_points_per_view: int,
    target_overlap_w2c: np.ndarray,
    inv_depth_power: float = 1.0,
    inv_depth_weight_clip_percentile: float = 0.98,
    inv_depth_min_depth: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    est_point_chunks = _extract_gaussian_alignment_points(
        gaussians,
        range(chunk_overlap),
        num_views=num_views,
        max_points_per_view=max_points_per_view,
    )
    ref_points = []
    est_points = []
    weights = []
    for overlap_idx, (source_idx, est_pts) in enumerate(zip(overlap_source, est_point_chunks)):
        if int(source_idx) not in alignment_points_by_source_idx:
            raise KeyError(f"Missing reference overlap points for source frame {source_idx}.")
        ref_pts = alignment_points_by_source_idx[int(source_idx)]
        n_pair = min(len(ref_pts), len(est_pts))
        if n_pair <= 0:
            continue
        ref_pts = ref_pts[:n_pair]
        est_pts = est_pts[:n_pair]
        ref_points.append(ref_pts)
        est_points.append(est_pts)
        depth = _point_depths_in_camera(ref_pts, target_overlap_w2c[overlap_idx])
        weights.append(
            _inverse_depth_weights(
                depth,
                power=inv_depth_power,
                clip_percentile=inv_depth_weight_clip_percentile,
                min_depth=inv_depth_min_depth,
            )
        )
    if not ref_points:
        raise ValueError("No valid overlap point pairs found.")
    return (
        np.concatenate(est_points, axis=0),
        np.concatenate(ref_points, axis=0),
        _positive_normalized_weights(np.concatenate(weights, axis=0), sum(len(x) for x in weights)),
    )


def _align_camera_and_points_umeyama_robust(
    source_points: np.ndarray,
    target_points: np.ndarray,
    *,
    source_w2c: np.ndarray,
    target_w2c: np.ndarray,
    weights: np.ndarray | None = None,
    camera_weight: float,
    point_weight: float,
    trim_percentile: float,
    max_iters: int,
    min_points: int,
) -> PointAlignmentResult:
    """Fit one Sim(3) to camera centers and overlap points with camera priority.

    Correspondence weights within each group are normalized to a fixed group total,
    then the camera/point group weights are applied. Robust trimming applies only to
    points; every valid camera pair is always used.
    """
    source = np.asarray(source_points, dtype=np.float32).reshape(-1, 3)
    target = np.asarray(target_points, dtype=np.float32).reshape(-1, 3)
    weight = _positive_normalized_weights(weights, source.shape[0])
    valid = (
        np.isfinite(source).all(axis=1)
        & np.isfinite(target).all(axis=1)
        & np.isfinite(weight)
        & (weight > 0.0)
    )
    source = source[valid]
    target = target[valid]
    if source.shape[0] < int(min_points):
        raise ValueError(f"Need at least {min_points} valid point pairs, got {source.shape[0]}.")
    weight = _positive_normalized_weights(weight[valid], source.shape[0])

    source_centers = _camera_centers_from_w2c(source_w2c)
    target_centers = _camera_centers_from_w2c(target_w2c)
    if source_centers.shape != target_centers.shape:
        raise ValueError(
            f"Camera-center shape mismatch: {source_centers.shape} vs {target_centers.shape}."
        )
    camera_valid = (
        np.isfinite(source_centers).all(axis=1)
        & np.isfinite(target_centers).all(axis=1)
    )
    source_centers = source_centers[camera_valid]
    target_centers = target_centers[camera_valid]
    if source_centers.shape[0] < 3:
        raise ValueError(
            f"Need at least 3 valid overlap camera centers, got {source_centers.shape[0]}."
        )

    camera_weight = float(camera_weight)
    point_weight = float(point_weight)
    if camera_weight <= 0.0 or point_weight <= 0.0:
        raise ValueError(
            f"Joint alignment weights must be positive, got camera={camera_weight}, "
            f"point={point_weight}."
        )

    def fit_joint(inlier_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        point_weights = _positive_normalized_weights(
            weight[inlier_mask],
            int(inlier_mask.sum()),
        )
        point_weights = point_weights / max(float(point_weights.sum()), 1e-12) * point_weight
        camera_weights = np.full(
            source_centers.shape[0],
            camera_weight / float(source_centers.shape[0]),
            dtype=np.float32,
        )
        joint_source = np.concatenate([source_centers, source[inlier_mask]], axis=0)
        joint_target = np.concatenate([target_centers, target[inlier_mask]], axis=0)
        joint_weights = np.concatenate([camera_weights, point_weights], axis=0)
        return _fit_sim3_points_umeyama(joint_source, joint_target, joint_weights)

    inlier = np.ones(source.shape[0], dtype=bool)
    r, t, s = fit_joint(inlier)
    trim_percentile = float(trim_percentile)
    max_iters = max(0, int(max_iters))
    for _ in range(max_iters):
        aligned = _apply_sim3_to_points(source, r, t, s)
        err = np.linalg.norm(aligned - target, axis=1)
        score = err / np.sqrt(np.maximum(weight, 1e-8))
        if 0.0 < trim_percentile < 1.0:
            threshold = float(np.quantile(score, trim_percentile))
            new_inlier = score <= threshold
        else:
            new_inlier = inlier
        if int(new_inlier.sum()) < int(min_points):
            break
        if np.array_equal(new_inlier, inlier):
            break
        inlier = new_inlier
        r, t, s = fit_joint(inlier)

    aligned = _apply_sim3_to_points(source, r, t, s)
    err = np.linalg.norm(aligned - target, axis=1)
    aligned_centers = _apply_sim3_to_points(source_centers, r, t, s)
    camera_err = np.linalg.norm(aligned_centers - target_centers, axis=1)
    weight = _positive_normalized_weights(weight, source.shape[0])
    return PointAlignmentResult(
        r=r.astype(np.float32),
        t=t.astype(np.float32),
        s=float(s),
        num_pairs=int(source.shape[0]),
        err_mean=float(err.mean()),
        err_p95=float(np.quantile(err, 0.95)),
        weighted_err_mean=_weighted_mean(err, weight),
        weight_mean=float(weight.mean()),
        weight_p95=float(np.quantile(weight, 0.95)),
        weight_max=float(weight.max()),
        weight_effective_pairs=_weighted_effective_count(weight, source.shape[0]),
        camera_err_mean=float(camera_err.mean()),
        camera_err_p95=float(np.quantile(camera_err, 0.95)),
    )


def _gaussians_to_cpu(gaussians):
    _ensure_da3_on_syspath()
    from depth_anything_3.specs import Gaussians  # type: ignore

    return Gaussians(
        means=gaussians.means.detach().cpu(),
        scales=gaussians.scales.detach().cpu(),
        rotations=gaussians.rotations.detach().cpu(),
        harmonics=gaussians.harmonics.detach().cpu(),
        opacities=gaussians.opacities.detach().cpu(),
    )


def _concat_gaussians(chunks: Sequence):
    _ensure_da3_on_syspath()
    from depth_anything_3.specs import Gaussians  # type: ignore

    if not chunks:
        raise ValueError("No gaussian chunks to concatenate.")
    return Gaussians(
        means=torch.cat([chunk.means for chunk in chunks], dim=1),
        scales=torch.cat([chunk.scales for chunk in chunks], dim=1),
        rotations=torch.cat([chunk.rotations for chunk in chunks], dim=1),
        harmonics=torch.cat([chunk.harmonics for chunk in chunks], dim=1),
        opacities=torch.cat([chunk.opacities for chunk in chunks], dim=1),
    )


def _mask_gaussians(gaussians, mask: np.ndarray):
    _ensure_da3_on_syspath()
    from depth_anything_3.specs import Gaussians  # type: ignore

    mask_t = torch.as_tensor(mask, device=gaussians.means.device, dtype=torch.bool)
    return Gaussians(
        means=gaussians.means[:, mask_t],
        scales=gaussians.scales[:, mask_t],
        rotations=gaussians.rotations[:, mask_t],
        harmonics=gaussians.harmonics[:, mask_t],
        opacities=gaussians.opacities[:, mask_t],
    )


def _gaussian_grid_export_mask(
    gaussians,
    depth: np.ndarray,
    *,
    prune_by_opacity_percentile: float | None,
    prune_border_gs: bool,
) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim != 3:
        raise ValueError(f"Expected depth with shape (V,H,W), got {depth.shape}.")

    src_v, out_h, out_w = depth.shape
    n_total = int(gaussians.means.shape[1])
    if n_total % src_v != 0:
        raise ValueError(f"Gaussian count {n_total} is not divisible by {src_v} views.")

    n_per_view = n_total // src_v
    if n_per_view != out_h * out_w:
        ratio = (out_h * out_w) / n_per_view
        dr = int(round(ratio**0.5))
        out_h = out_h // dr
        out_w = out_w // dr
        if out_h * out_w != n_per_view:
            raise ValueError(
                f"Cannot infer GS grid shape: {n_per_view} != {out_h} * {out_w}."
            )

    device = gaussians.means.device
    if prune_border_gs:
        mask = torch.zeros((src_v, out_h, out_w), device=device, dtype=torch.bool)
        trim_h = int(8 / 256 * out_h)
        trim_w = int(8 / 256 * out_w)
        h0 = trim_h
        h1 = out_h - trim_h if trim_h > 0 else out_h
        w0 = trim_w
        w1 = out_w - trim_w if trim_w > 0 else out_w
        mask[:, h0:h1, w0:w1] = True
    else:
        mask = torch.ones((src_v, out_h, out_w), device=device, dtype=torch.bool)

    if (
        prune_by_opacity_percentile is not None
        and 0.0 < float(prune_by_opacity_percentile) < 1.0
    ):
        op = gaussians.opacities[0]
        if op.ndim > 1 and op.shape[-1] == 1:
            op = op[..., 0]
        op_map = op.reshape(src_v, out_h, out_w)
        if mask.any():
            flat = op_map[mask].float().flatten()
            n = flat.numel()
            k = max(1, min(n, int(n * float(prune_by_opacity_percentile))))
            threshold = torch.kthvalue(flat, k).values
            mask = mask & (op_map >= threshold)

    return mask.reshape(-1).detach().cpu().numpy()


def _positions_in_frustum_torch(
    positions: torch.Tensor,
    w2c: torch.Tensor,
    intrinsics: torch.Tensor,
    image_width: int,
    image_height: int,
    *,
    near: float = 0.01,
    far: float = 1e6,
) -> torch.Tensor:
    pos_h = torch.cat([positions, torch.ones_like(positions[:, :1])], dim=-1)
    pos_cam = torch.einsum("pk,cjk->pcj", pos_h, w2c)
    x, y, z = pos_cam[..., 0], pos_cam[..., 1], pos_cam[..., 2]

    in_front = z > near
    in_range = z < far
    z_safe = z.clamp_min(1e-6)

    fx = intrinsics[:, 0, 0]
    fy = intrinsics[:, 1, 1]
    cx = intrinsics[:, 0, 2]
    cy = intrinsics[:, 1, 2]
    u = fx * (x / z_safe) + cx
    v = fy * (y / z_safe) + cy
    in_image = (u >= 0) & (u < image_width) & (v >= 0) & (v < image_height)
    return in_front & in_range & in_image


def _compute_frustum_ownership_mask(
    chunk_idx: int,
    positions: np.ndarray,
    chunk_w2c: Sequence[np.ndarray],
    chunk_intrinsics: Sequence[np.ndarray],
    image_shape: tuple[int, int],
    *,
    max_diff_m: float,
    device: torch.device,
    batch_size: int,
    oob_other: float = 1e6,
    oob_owned: float = 1e5,
) -> np.ndarray:
    n_chunks = len(chunk_w2c)
    pos = np.asarray(positions, dtype=np.float32).reshape(-1, 3)
    if n_chunks <= 1 or len(pos) == 0:
        return np.ones(len(pos), dtype=bool)

    batch_size = max(1, int(batch_size))
    dtype = torch.float32
    image_h, image_w = int(image_shape[0]), int(image_shape[1])
    w2c_list = [
        torch.from_numpy(_pad_to_44(np.asarray(w2c, dtype=np.float32))).to(
            device=device, dtype=dtype
        )
        for w2c in chunk_w2c
    ]
    c2w_list = [torch.linalg.inv(w2c) for w2c in w2c_list]
    k_list = [
        torch.from_numpy(np.asarray(k, dtype=np.float32)).to(device=device, dtype=dtype)
        for k in chunk_intrinsics
    ]

    keep_parts: List[np.ndarray] = []
    for start in range(0, len(pos), batch_size):
        end = min(start + batch_size, len(pos))
        pos_t = torch.from_numpy(pos[start:end]).to(device=device, dtype=dtype)

        all_distances: List[torch.Tensor] = []
        for other_idx, (w2c, c2w, k) in enumerate(zip(w2c_list, c2w_list, k_list)):
            oob = oob_other if other_idx != chunk_idx else oob_owned
            centers = c2w[:, :3, 3]
            center_dist = torch.linalg.norm(pos_t[:, None, :] - centers[None, :, :], dim=-1)
            in_frustum = _positions_in_frustum_torch(
                pos_t,
                w2c,
                k,
                image_width=image_w,
                image_height=image_h,
            )
            center_dist = torch.where(
                in_frustum,
                center_dist,
                torch.full_like(center_dist, float(oob)),
            )
            all_distances.append(center_dist.min(dim=1).values)

        dists = torch.stack(all_distances, dim=0)
        sorted_dists, sorted_idx = torch.sort(dists, dim=0)
        inds0 = sorted_idx[0]
        inds1 = sorted_idx[1]
        dists0 = sorted_dists[0]
        dists1 = sorted_dists[1]
        keep = (inds0 == chunk_idx) | (
            (inds1 == chunk_idx) & (dists1 - dists0 < float(max_diff_m))
        )
        keep_parts.append(keep.detach().cpu().numpy())

    return np.concatenate(keep_parts, axis=0)


def _export_gaussians_ply_direct(gaussians, save_path: Path) -> None:
    _ensure_da3_on_syspath()
    from depth_anything_3.utils.gsply_helpers import export_ply, inverse_sigmoid  # type: ignore

    opacities = inverse_sigmoid(gaussians.opacities.clamp(1e-6, 1.0 - 1e-6))[0]
    if opacities.ndim > 1 and opacities.shape[-1] == 1:
        opacities = opacities[..., 0]

    export_ply(
        means=gaussians.means[0],
        scales=gaussians.scales[0],
        rotations=gaussians.rotations[0],
        harmonics=gaussians.harmonics[0],
        opacities=opacities,
        path=Path(save_path),
        shift_and_scale=False,
        save_sh_dc_only=True,
    )


def _estimate_sampled_render_fps(source_frame_indices: np.ndarray, source_fps: float) -> float:
    if source_frame_indices.size <= 1:
        return float(max(1.0, round(source_fps)))
    span_frames = int(source_frame_indices[-1] - source_frame_indices[0] + 1)
    duration_sec = max(float(span_frames) / float(max(source_fps, 1e-6)), 1e-6)
    return float(max(1.0, source_frame_indices.size / duration_sec))


def _interpolate_intrinsics(
    k_keyframes: np.ndarray,
    key_indices: Sequence[int],
    n_total: int,
) -> np.ndarray:
    k_keyframes = np.asarray(k_keyframes, dtype=np.float32)
    if len(key_indices) == 0:
        raise ValueError("No key_indices provided for intrinsics interpolation.")
    if len(key_indices) != int(k_keyframes.shape[0]):
        raise ValueError(
            f"Intrinsics/key index count mismatch: {k_keyframes.shape[0]} vs {len(key_indices)}"
        )
    if len(key_indices) == 1:
        return np.repeat(k_keyframes[:1], int(n_total), axis=0).astype(np.float32)

    times_key = np.asarray(key_indices, dtype=np.float64)
    times_all = np.arange(int(n_total), dtype=np.float64)
    times_clamped = np.clip(times_all, times_key[0], times_key[-1])
    flat = k_keyframes.reshape(k_keyframes.shape[0], -1).astype(np.float64)
    flat_interp = np.column_stack(
        [np.interp(times_clamped, times_key, flat[:, dim]) for dim in range(flat.shape[1])]
    )
    return flat_interp.reshape(int(n_total), *k_keyframes.shape[1:]).astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chunked VIPE pose estimation + DA3 Gaussian reconstruction."
    )
    parser.add_argument("--input_video_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--force", action="store_true")

    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--vipe_overrides", type=str, nargs="+", default=None)
    parser.add_argument("--vipe_full_mode", action="store_true")

    parser.add_argument(
        "--max_frames",
        type=int,
        default=0,
        help="Limit the raw source-video frame range before uniform sampling.",
    )
    parser.add_argument(
        "--num_chunks",
        type=int,
        default=None,
        help="Number of chunks for uniform sampling. Inferred from --chunk_split_frames when set.",
    )
    parser.add_argument("--chunk_size", type=int, default=128)
    parser.add_argument("--chunk_overlap", type=int, default=10)
    parser.add_argument(
        "--chunk_point_align_max_points_per_frame",
        type=int,
        default=4096,
        help="Max same-pixel GS point pairs sampled per overlap frame for point alignment.",
    )
    parser.add_argument(
        "--chunk_point_align_trim_percentile",
        type=float,
        default=0.8,
        help="Robust point alignment keeps this residual percentile on each refit iteration.",
    )
    parser.add_argument(
        "--chunk_point_align_max_iters",
        type=int,
        default=3,
        help="Robust point alignment refit iterations.",
    )
    parser.add_argument(
        "--chunk_point_align_min_points",
        type=int,
        default=1000,
        help="Minimum valid overlap point pairs required before falling back to camera alignment.",
    )
    parser.add_argument(
        "--chunk_point_align_inv_depth_power",
        type=float,
        default=1.0,
        help="Power used for inverse-depth point-alignment weights.",
    )
    parser.add_argument(
        "--chunk_point_align_inv_depth_weight_clip_percentile",
        type=float,
        default=0.98,
        help="Clip inverse-depth weights at this percentile before normalization.",
    )
    parser.add_argument(
        "--chunk_point_align_inv_depth_min_depth",
        type=float,
        default=1e-3,
        help="Minimum positive camera depth used when computing inverse-depth weights.",
    )
    parser.add_argument(
        "--chunk_split_frames",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Optional source-video frame ids where adjacent chunks should split. "
            "The next chunk reuses the previous chunk's last overlap sampled frames, "
            "then samples the remaining views from its own split segment."
        ),
    )

    parser.add_argument(
        "--da3_model_name",
        type=str,
        default="depth-anything/DA3NESTED-GIANT-LARGE-1.1",
    )
    parser.add_argument(
        "--da3_model_path_custom",
        type=str,
        default=str(DEFAULT_RECON_DA3_MODEL_PATH),
    )
    parser.add_argument("--da3_process_res", type=int, default=None)
    parser.add_argument(
        "--da3_process_method",
        type=str,
        default="upper_bound_resize",
        choices=["upper_bound_resize", "lower_bound_resize"],
    )
    parser.add_argument(
        "--max_resolution",
        type=int,
        default=0,
        help="If > 0, use as DA3 short-side cap via lower_bound_resize.",
    )

    parser.add_argument("--gs_down_ratio", type=int, default=2)
    parser.add_argument("--gs_scale_extra_multiplier", type=float, default=1.0)
    parser.add_argument("--gs_ply_prune_opacity_percentile", type=float, default=None)
    parser.add_argument(
        "--gs_frustum_ownership_max_diff_m",
        type=float,
        default=1.0,
        help=(
            "Ownership pruning tolerance. If the current chunk is second closest, "
            "keep the Gaussian only when the distance gap is below this value."
        ),
    )
    parser.add_argument(
        "--gs_frustum_ownership_batch_size",
        type=int,
        default=10240,
        help="Number of Gaussian centers per ownership-pruning batch.",
    )
    parser.add_argument(
        "--no_gs_ds_feature_mode",
        dest="gs_ds_feature_mode",
        action="store_false",
        help="Disable the default release-friendly GS feature downsampling mode.",
    )
    parser.set_defaults(gs_ds_feature_mode=True)

    parser.add_argument("--render_fps", type=float, default=None)
    parser.add_argument("--render_chunk_size", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_video = Path(args.input_video_path).expanduser().resolve()
    if not input_video.is_file():
        raise FileNotFoundError(f"Input video not found: {input_video}")

    output_dir = _build_chunked_output_dir(str(input_video), args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    done_marker = output_dir / ".done"
    if done_marker.is_file() and not args.force:
        print(
            f"[vipe_da3_chunked_gs] Skipping {input_video.name}: {done_marker} exists. "
            "Use --force to re-run."
        )
        return

    frame_count, fps = _probe_video(str(input_video))
    source_total = frame_count
    if args.max_frames > 0:
        source_total = min(source_total, int(args.max_frames))

    split_frames = [int(frame) for frame in args.chunk_split_frames or []]
    if split_frames:
        num_chunks = len(split_frames) + 1
        if args.num_chunks is not None and int(args.num_chunks) != num_chunks:
            raise ValueError(
                f"--chunk_split_frames={split_frames} implies {num_chunks} chunks, "
                f"but --num_chunks={args.num_chunks}."
            )
    else:
        num_chunks = int(args.num_chunks) if args.num_chunks is not None else 3

    chunk_size = int(args.chunk_size)
    chunk_overlap = int(args.chunk_overlap)
    target_unique = _target_unique_views(num_chunks, chunk_size, chunk_overlap)
    if source_total < target_unique:
        raise ValueError(
            f"Need at least {target_unique} source frames for "
            f"{num_chunks} chunks of {chunk_size} with overlap {chunk_overlap}, "
            f"but video/max_frames provides {source_total}."
        )

    if split_frames:
        sampled_abs, source_indices_by_chunk = _build_split_chunk_source_indices(
            source_total=source_total,
            split_frames=split_frames,
            chunk_size=chunk_size,
            overlap=chunk_overlap,
        )
        sampling_mode = "split"
    else:
        sampled_abs, source_indices_by_chunk = _build_uniform_chunk_source_indices(
            source_total=source_total,
            num_chunks=num_chunks,
            chunk_size=chunk_size,
            overlap=chunk_overlap,
        )
        sampling_mode = "uniform"
    chunk_specs = _build_chunk_specs_from_source_indices(
        source_indices_by_chunk,
        sampled_abs,
        chunk_overlap,
    )

    print(f"[vipe_da3_chunked_gs] Input video: {input_video}")
    print(f"[vipe_da3_chunked_gs] Output dir:  {output_dir}")
    print(
        f"[vipe_da3_chunked_gs] Sampling {len(sampled_abs)} unique frames "
        f"({num_chunks} chunks x {chunk_size}, overlap={chunk_overlap}, mode={sampling_mode}) "
        f"from {source_total} source frames."
    )
    if split_frames:
        print(f"[vipe_da3_chunked_gs] Chunk split frames: {split_frames}")
    print(
        f"[vipe_da3_chunked_gs] Alignment mode: {CHUNK_ALIGNMENT_MODE} "
        f"(camera:point={JOINT_CAMERA_WEIGHT:g}:{JOINT_POINT_WEIGHT:g}, "
        f"point weights={CHUNK_POINT_ALIGN_WEIGHT_MODE}); "
        f"overlap merge mode: {CHUNK_OVERLAP_MERGE_MODE}"
    )

    images_sampled = _read_video_frames_rgb(str(input_video), sampled_abs.astype(np.int64).tolist())
    if len(images_sampled) != len(sampled_abs):
        raise RuntimeError(f"Expected {len(sampled_abs)} sampled images, got {len(images_sampled)}.")
    sampled_fps = _estimate_sampled_render_fps(sampled_abs, fps)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[vipe_da3_chunked_gs] Device:      {device}")
    print(f"[vipe_da3_chunked_gs] Source fps={fps:.4g}, sampled fps={sampled_fps:.4g}")

    da3_model_path_custom = None
    if args.da3_model_path_custom:
        da3_model_path_custom = str(Path(args.da3_model_path_custom).expanduser().resolve())
        if not Path(da3_model_path_custom).is_file():
            raise FileNotFoundError(f"DA3 checkpoint not found: {da3_model_path_custom}")
        print(f"[vipe_da3_chunked_gs] DA3 ckpt:    {da3_model_path_custom}")

    print("[vipe_da3_chunked_gs] Loading DA3 model...")
    da3_model = load_da3_model(
        da3_model_name=args.da3_model_name,
        da3_model_path_custom=da3_model_path_custom,
        device=str(device),
    )
    da3_model.eval()

    if args.da3_process_res is not None:
        da3_process_res = int(args.da3_process_res)
        da3_process_method = str(args.da3_process_method)
    elif int(args.max_resolution) > 0:
        da3_process_res = int(args.max_resolution)
        da3_process_method = "lower_bound_resize"
    else:
        h0, w0 = images_sampled[0].shape[:2]
        da3_process_res = int(max(h0, w0))
        da3_process_method = "upper_bound_resize"

    VIPE = _import_vipe_class()
    frames_np = np.stack(images_sampled, axis=0).astype(np.float32) / 255.0
    frames_thwc = torch.from_numpy(frames_np).contiguous()

    with tempfile.TemporaryDirectory(prefix="vipe_da3_chunked_gs_") as tmpdir:
        vipe_output_path = Path(tmpdir) / "vipe_out"
        vipe_output_path.mkdir(parents=True, exist_ok=True)
        vipe_overrides = args.vipe_overrides or _vipe_default_overrides(vipe_output_path)

        print("[vipe_da3_chunked_gs] Loading VIPE...")
        vipe_kwargs = {"fast_mode": not bool(args.vipe_full_mode)}
        vipe = VIPE(vipe_overrides, **vipe_kwargs)

        print("[vipe_da3_chunked_gs] Running VIPE on sampled frames...")
        vipe_out = vipe.infer_frames(frames_thwc, fps=float(sampled_fps), name=input_video.stem)

        c2w = vipe_out.extrinsics_c2w.to(dtype=torch.float32)
        w2c = torch.linalg.inv(c2w)
        intrinsics_vipe = _intrinsics_vec_to_k33(vipe_out.intrinsics.to(dtype=torch.float32))

        w2c_np_sampled = w2c.cpu().numpy().astype(np.float32)
        k_np_sampled = intrinsics_vipe.cpu().numpy().astype(np.float32)

        np.savez(
            output_dir / "vipe_predictions.npz",
            frame_ids=vipe_out.frame_ids.cpu().numpy().astype(np.int64),
            w2c_vipe_sampled=w2c_np_sampled,
            intrinsics_vipe_sampled=k_np_sampled,
            indices_sampled=sampled_abs,
            fps=np.asarray([sampled_fps], dtype=np.float32),
            source_fps=np.asarray([fps], dtype=np.float32),
            sampling_mode=np.asarray([sampling_mode]),
            chunk_split_frames=np.asarray(split_frames, dtype=np.int64),
            input_video_path=np.asarray([str(input_video)]),
        )

        _ensure_da3_on_syspath()
        from depth_anything_3.utils.pose_align import align_poses_umeyama  # type: ignore

        gaussian_chunks = []
        depth_chunks = []
        render_w2c_chunks = []
        render_k_chunks = []
        render_source_index_chunks = []
        global_w2c_by_source_idx: dict[int, np.ndarray] = {}
        alignment_points_by_source_idx: dict[int, np.ndarray] = {}

        chunk_raw_w2c = []
        chunk_local_aligned_w2c = []
        chunk_global_aligned_w2c = []
        chunk_sim3_r = []
        chunk_sim3_t = []
        chunk_sim3_s = []
        chunk_alignment_method_used = []
        chunk_point_align_num_pairs = []
        chunk_point_align_err_mean = []
        chunk_point_align_err_p95 = []
        chunk_point_align_weighted_err_mean = []
        chunk_point_align_weight_mean = []
        chunk_point_align_weight_p95 = []
        chunk_point_align_weight_max = []
        chunk_point_align_weight_effective_pairs = []

        for spec in chunk_specs:
            chunk_sampled = spec.sampled_indices
            trajectory_keep_local = spec.keep_local_indices
            gs_keep_local = _chunk_gs_keep_indices(
                spec.chunk_idx,
                len(chunk_sampled),
                chunk_overlap,
            )
            images_chunk = [images_sampled[idx] for idx in chunk_sampled]
            w2c_chunk = w2c_np_sampled[chunk_sampled]
            k_chunk = k_np_sampled[chunk_sampled]

            print(
                f"[vipe_da3_chunked_gs] DA3 chunk {spec.chunk_idx + 1}/{len(chunk_specs)}: "
                f"source_span={spec.source_indices[0]}..{spec.source_indices[-1]} "
                f"gs_keep={gs_keep_local[0]}:{gs_keep_local[-1] + 1} "
                f"trj_keep={trajectory_keep_local[0]}:{trajectory_keep_local[-1] + 1} "
                f"process_res={da3_process_res}"
            )
            pred = da3_model.inference(
                image=images_chunk,
                extrinsics=w2c_chunk,
                intrinsics=k_chunk,
                align_to_input_extrinsics=False,
                align_to_input_ext_scale=False,
                infer_gs=True,
                process_res=da3_process_res,
                process_res_method=da3_process_method,
                export_dir=None,
                export_format="mini_npz",
                use_aligned_pred_cam=True,
                gs_down_ratio=args.gs_down_ratio,
                gs_scale_extra_multiplier=args.gs_scale_extra_multiplier,
                gs_ds_feature_mode=args.gs_ds_feature_mode,
            )

            if pred.gaussians is None:
                raise RuntimeError(f"DA3 chunk {spec.chunk_idx} did not return gaussians.")
            if pred.extrinsics is None:
                raise RuntimeError(f"DA3 chunk {spec.chunk_idx} did not return predicted poses.")

            raw_pred_w2c = _pad_to_44(np.asarray(pred.extrinsics, dtype=np.float32))
            local_aligned_w2c = _compute_aligned_pred_w2c(raw_pred_w2c, w2c_chunk)

            if spec.chunk_idx == 0:
                r = np.eye(3, dtype=np.float32)
                t = np.zeros(3, dtype=np.float32)
                s = 1.0
                global_aligned_w2c = local_aligned_w2c
                aligned_gaussians = pred.gaussians
                alignment_method_used = "identity"
                point_result = None
            else:
                overlap_source = spec.source_indices[:chunk_overlap]
                ref_overlap = np.stack(
                    [global_w2c_by_source_idx[idx] for idx in overlap_source],
                    axis=0,
                )
                est_overlap = local_aligned_w2c[:chunk_overlap]

                point_result = None
                r_cam, t_cam, s_cam = align_poses_umeyama(
                    ref_overlap,
                    est_overlap,
                    return_aligned=False,
                )
                r, t, s = r_cam, t_cam, s_cam
                alignment_method_used = "camera"

                try:
                    est_points, ref_points, point_weights = _collect_overlap_point_pairs(
                        pred.gaussians,
                        overlap_source,
                        alignment_points_by_source_idx,
                        chunk_overlap=chunk_overlap,
                        num_views=len(chunk_sampled),
                        max_points_per_view=args.chunk_point_align_max_points_per_frame,
                        target_overlap_w2c=ref_overlap,
                        inv_depth_power=args.chunk_point_align_inv_depth_power,
                        inv_depth_weight_clip_percentile=(
                            args.chunk_point_align_inv_depth_weight_clip_percentile
                        ),
                        inv_depth_min_depth=args.chunk_point_align_inv_depth_min_depth,
                    )
                    point_result = _align_camera_and_points_umeyama_robust(
                        est_points,
                        ref_points,
                        source_w2c=est_overlap,
                        target_w2c=ref_overlap,
                        weights=point_weights,
                        camera_weight=JOINT_CAMERA_WEIGHT,
                        point_weight=JOINT_POINT_WEIGHT,
                        trim_percentile=float(args.chunk_point_align_trim_percentile),
                        max_iters=int(args.chunk_point_align_max_iters),
                        min_points=int(args.chunk_point_align_min_points),
                    )
                    r = point_result.r
                    t = point_result.t
                    s = point_result.s
                    alignment_method_used = "camera_priority_joint"
                    print(
                        f"[vipe_da3_chunked_gs] Chunk {spec.chunk_idx + 1} joint Sim(3): "
                        f"scale={s:.6g}, camera_only_scale={s_cam:.6g}, "
                        f"camera:point={JOINT_CAMERA_WEIGHT:g}:{JOINT_POINT_WEIGHT:g}"
                    )
                except Exception as exc:
                    print(
                        f"[vipe_da3_chunked_gs] Warning: point alignment failed for "
                        f"chunk {spec.chunk_idx + 1}; falling back to camera alignment. "
                        f"Reason: {exc}"
                    )
                    r, t, s = r_cam, t_cam, s_cam

                r = r.astype(np.float32)
                t = t.astype(np.float32)
                s = float(s)
                global_aligned_w2c = _apply_sim3_to_w2c(local_aligned_w2c, r, t, s)
                aligned_gaussians = _apply_sim3_to_gaussians(pred.gaussians, r, t, s)
                overlap_err = np.linalg.norm(
                    np.linalg.inv(global_aligned_w2c[:chunk_overlap])[:, :3, 3]
                    - np.linalg.inv(ref_overlap)[:, :3, 3],
                    axis=1,
                )
                print(
                    f"[vipe_da3_chunked_gs] Chunk {spec.chunk_idx + 1} alignment: "
                    f"method={alignment_method_used}, scale={s:.6g}, "
                    f"overlap_pos_err_mean={overlap_err.mean():.6g}, max={overlap_err.max():.6g}"
                )
                if point_result is not None:
                    print(
                        f"[vipe_da3_chunked_gs] Chunk {spec.chunk_idx + 1} point alignment: "
                        f"pairs={point_result.num_pairs:,}, "
                        f"point_err_mean={point_result.err_mean:.6g}, "
                        f"p95={point_result.err_p95:.6g}, "
                        f"weighted_mean={point_result.weighted_err_mean:.6g}, "
                        f"camera_err_mean={point_result.camera_err_mean:.6g}, "
                        f"camera_p95={point_result.camera_err_p95:.6g}"
                    )

            for source_idx, pose in zip(spec.source_indices, global_aligned_w2c):
                if source_idx not in global_w2c_by_source_idx:
                    global_w2c_by_source_idx[source_idx] = pose

            if spec.chunk_idx < len(chunk_specs) - 1:
                tail_local = list(range(len(chunk_sampled) - chunk_overlap, len(chunk_sampled)))
                tail_points = _extract_gaussian_alignment_points(
                    aligned_gaussians,
                    tail_local,
                    num_views=len(chunk_sampled),
                    max_points_per_view=args.chunk_point_align_max_points_per_frame,
                )
                for source_idx, points in zip(spec.source_indices[-chunk_overlap:], tail_points):
                    alignment_points_by_source_idx[int(source_idx)] = points

            kept_gaussians = _select_gaussian_views(
                aligned_gaussians,
                gs_keep_local,
                num_views=len(chunk_sampled),
            )
            gaussian_chunks.append(_gaussians_to_cpu(kept_gaussians))
            depth_chunks.append(np.asarray(pred.depth, dtype=np.float32)[gs_keep_local])
            render_w2c_chunks.append(global_aligned_w2c[trajectory_keep_local])
            render_k_chunks.append(k_chunk[trajectory_keep_local])
            render_source_index_chunks.append(
                np.asarray(spec.kept_source_indices, dtype=np.int64)
            )

            chunk_raw_w2c.append(raw_pred_w2c)
            chunk_local_aligned_w2c.append(local_aligned_w2c)
            chunk_global_aligned_w2c.append(global_aligned_w2c)
            chunk_sim3_r.append(r)
            chunk_sim3_t.append(t)
            chunk_sim3_s.append(s)
            chunk_alignment_method_used.append(alignment_method_used)
            chunk_point_align_num_pairs.append(
                0 if point_result is None else point_result.num_pairs
            )
            chunk_point_align_err_mean.append(
                np.nan if point_result is None else point_result.err_mean
            )
            chunk_point_align_err_p95.append(
                np.nan if point_result is None else point_result.err_p95
            )
            chunk_point_align_weighted_err_mean.append(
                np.nan if point_result is None else point_result.weighted_err_mean
            )
            chunk_point_align_weight_mean.append(
                np.nan if point_result is None else point_result.weight_mean
            )
            chunk_point_align_weight_p95.append(
                np.nan if point_result is None else point_result.weight_p95
            )
            chunk_point_align_weight_max.append(
                np.nan if point_result is None else point_result.weight_max
            )
            chunk_point_align_weight_effective_pairs.append(
                np.nan if point_result is None else point_result.weight_effective_pairs
            )
            del pred, aligned_gaussians, kept_gaussians
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        frustum_prune_before_counts: List[int] = []
        frustum_prune_after_grid_counts: List[int] = []
        frustum_prune_after_ownership_counts: List[int] = []
        prune_device = (
            device
            if device.type == "cuda" and torch.cuda.is_available()
            else torch.device("cpu")
        )
        prune_border_gs = not (
            args.gs_ply_prune_opacity_percentile is not None
            and args.gs_ply_prune_opacity_percentile > 0
        )
        image_shape = images_sampled[0].shape[:2]
        print(
            "[vipe_da3_chunked_gs] Pruning GS by frustum ownership "
            f"(max_diff_m={args.gs_frustum_ownership_max_diff_m}, "
            f"device={prune_device}, batch_size={args.gs_frustum_ownership_batch_size})..."
        )
        pruned_gaussian_chunks = []
        for chunk_idx, (gaussian_chunk, depth_chunk) in enumerate(
            zip(gaussian_chunks, depth_chunks)
        ):
            n_before = int(gaussian_chunk.means.shape[1])
            grid_mask = _gaussian_grid_export_mask(
                gaussian_chunk,
                depth_chunk,
                prune_by_opacity_percentile=args.gs_ply_prune_opacity_percentile,
                prune_border_gs=prune_border_gs,
            )
            gaussian_chunk = _mask_gaussians(gaussian_chunk, grid_mask)
            n_after_grid = int(gaussian_chunk.means.shape[1])

            ownership_mask = _compute_frustum_ownership_mask(
                chunk_idx,
                gaussian_chunk.means[0].detach().cpu().numpy(),
                render_w2c_chunks,
                render_k_chunks,
                image_shape,
                max_diff_m=float(args.gs_frustum_ownership_max_diff_m),
                device=prune_device,
                batch_size=int(args.gs_frustum_ownership_batch_size),
            )
            gaussian_chunk = _mask_gaussians(gaussian_chunk, ownership_mask)
            n_after_ownership = int(gaussian_chunk.means.shape[1])
            pruned_gaussian_chunks.append(gaussian_chunk)

            frustum_prune_before_counts.append(n_before)
            frustum_prune_after_grid_counts.append(n_after_grid)
            frustum_prune_after_ownership_counts.append(n_after_ownership)
            pct = 100.0 * n_after_ownership / n_before if n_before > 0 else 0.0
            print(
                f"[vipe_da3_chunked_gs] Chunk {chunk_idx + 1} prune: "
                f"{n_before:,} -> {n_after_grid:,} after grid/export mask -> "
                f"{n_after_ownership:,} after ownership ({pct:.1f}%)"
            )

        gaussian_chunks = pruned_gaussian_chunks
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        final_gaussians = _concat_gaussians(gaussian_chunks)
        w2c_recon_sampled = np.concatenate(render_w2c_chunks, axis=0).astype(np.float32)
        k_recon_sampled = np.concatenate(render_k_chunks, axis=0).astype(np.float32)
        source_indices_recon_sampled = np.concatenate(render_source_index_chunks, axis=0).astype(
            np.int64
        )

        w2c_render = _interpolate_w2c(
            w2c_recon_sampled,
            source_indices_recon_sampled.tolist(),
            source_total,
        )
        k_render = _interpolate_intrinsics(
            k_recon_sampled,
            source_indices_recon_sampled.tolist(),
            source_total,
        )
        source_indices_render = np.arange(source_total, dtype=np.int64)
        render_fps = float(args.render_fps) if args.render_fps is not None else float(fps)

        final_ply_path = output_dir / "reconstructed_scene.ply"
        _export_gaussians_ply_direct(final_gaussians, final_ply_path)
        print(f"[vipe_da3_chunked_gs] Saved PLY to {final_ply_path}")

        np.savez(
            output_dir / "cameras.npz",
            w2c_render=w2c_render,
            intrinsics_render=k_render,
            source_indices_render=source_indices_render,
            w2c_recon_sampled=w2c_recon_sampled,
            intrinsics_recon_sampled=k_recon_sampled,
            source_indices_recon_sampled=source_indices_recon_sampled,
            indices_sampled=sampled_abs,
            w2c_vipe_sampled=w2c_np_sampled,
            intrinsics_vipe_sampled=k_np_sampled,
            chunk_source_spans=np.asarray(
                [[spec.source_indices[0], spec.source_indices[-1]] for spec in chunk_specs],
                dtype=np.int64,
            ),
            chunk_source_indices=np.asarray(
                [spec.source_indices for spec in chunk_specs],
                dtype=np.int64,
            ),
            chunk_sampled_indices=np.asarray(
                [spec.sampled_indices for spec in chunk_specs],
                dtype=np.int64,
            ),
            chunk_keep_local_starts=np.asarray(
                [spec.keep_local_start for spec in chunk_specs],
                dtype=np.int64,
            ),
            chunk_raw_pred_w2c=np.stack(chunk_raw_w2c, axis=0),
            chunk_local_aligned_w2c=np.stack(chunk_local_aligned_w2c, axis=0),
            chunk_global_aligned_w2c=np.stack(chunk_global_aligned_w2c, axis=0),
            chunk_sim3_r=np.stack(chunk_sim3_r, axis=0),
            chunk_sim3_t=np.stack(chunk_sim3_t, axis=0),
            chunk_sim3_s=np.asarray(chunk_sim3_s, dtype=np.float32),
            chunk_alignment_mode=np.asarray([CHUNK_ALIGNMENT_MODE]),
            chunk_alignment_method_used=np.asarray(chunk_alignment_method_used),
            chunk_joint_camera_weight=np.asarray([JOINT_CAMERA_WEIGHT], dtype=np.float32),
            chunk_joint_point_weight=np.asarray([JOINT_POINT_WEIGHT], dtype=np.float32),
            chunk_point_align_num_pairs=np.asarray(chunk_point_align_num_pairs, dtype=np.int64),
            chunk_point_align_err_mean=np.asarray(chunk_point_align_err_mean, dtype=np.float32),
            chunk_point_align_err_p95=np.asarray(chunk_point_align_err_p95, dtype=np.float32),
            chunk_point_align_weighted_err_mean=np.asarray(
                chunk_point_align_weighted_err_mean,
                dtype=np.float32,
            ),
            chunk_point_align_weight_mean=np.asarray(
                chunk_point_align_weight_mean,
                dtype=np.float32,
            ),
            chunk_point_align_weight_p95=np.asarray(
                chunk_point_align_weight_p95,
                dtype=np.float32,
            ),
            chunk_point_align_weight_max=np.asarray(
                chunk_point_align_weight_max,
                dtype=np.float32,
            ),
            chunk_point_align_weight_effective_pairs=np.asarray(
                chunk_point_align_weight_effective_pairs,
                dtype=np.float32,
            ),
            chunk_point_align_weight_mode=np.asarray([CHUNK_POINT_ALIGN_WEIGHT_MODE]),
            chunk_point_align_inv_depth_power=np.asarray(
                [float(args.chunk_point_align_inv_depth_power)],
                dtype=np.float32,
            ),
            chunk_point_align_inv_depth_weight_clip_percentile=np.asarray(
                [float(args.chunk_point_align_inv_depth_weight_clip_percentile)],
                dtype=np.float32,
            ),
            chunk_overlap_merge_mode=np.asarray([CHUNK_OVERLAP_MERGE_MODE]),
            gs_frustum_ownership_prune=np.asarray(
                [GS_FRUSTUM_OWNERSHIP_PRUNE],
                dtype=bool,
            ),
            gs_frustum_ownership_max_diff_m=np.asarray(
                [float(args.gs_frustum_ownership_max_diff_m)],
                dtype=np.float32,
            ),
            gs_frustum_prune_before_counts=np.asarray(
                frustum_prune_before_counts,
                dtype=np.int64,
            ),
            gs_frustum_prune_after_grid_counts=np.asarray(
                frustum_prune_after_grid_counts,
                dtype=np.int64,
            ),
            gs_frustum_prune_after_ownership_counts=np.asarray(
                frustum_prune_after_ownership_counts,
                dtype=np.int64,
            ),
            fps=np.asarray([sampled_fps], dtype=np.float32),
            source_fps=np.asarray([fps], dtype=np.float32),
            render_fps=np.asarray([render_fps], dtype=np.float32),
            render_cadence=np.asarray([RENDER_CADENCE]),
            sampling_mode=np.asarray([sampling_mode]),
            chunk_split_frames=np.asarray(split_frames, dtype=np.int64),
            num_chunks=np.asarray([num_chunks], dtype=np.int32),
            chunk_size=np.asarray([chunk_size], dtype=np.int32),
            chunk_overlap=np.asarray([chunk_overlap], dtype=np.int32),
        )

        from depth_anything_3.model.utils.gs_renderer import run_renderer_in_chunk_w_trj_mode  # type: ignore

        gs_device = device
        if hasattr(da3_model, "model"):
            try:
                gs_device = next(da3_model.model.parameters()).device
            except StopIteration:
                gs_device = device

        gaussians = _load_gaussian_ply_to_gaussians(str(final_ply_path), device=gs_device)
        render_extr = torch.from_numpy(w2c_render).to(device=gs_device, dtype=gaussians.means.dtype)[None]
        render_intr = torch.from_numpy(k_render).to(device=gs_device, dtype=gaussians.means.dtype)[None]
        if render_extr.shape[-2:] == (3, 4):
            pad = torch.tensor([0, 0, 0, 1], device=gs_device, dtype=gaussians.means.dtype).view(
                1,
                1,
                1,
                4,
            )
            render_extr = torch.cat(
                [render_extr, pad.expand(render_extr.shape[0], render_extr.shape[1], -1, -1)],
                dim=-2,
            )

        render_h, render_w = images_sampled[0].shape[:2]
        print(
            f"[vipe_da3_chunked_gs] Rendering {render_extr.shape[1]} {RENDER_CADENCE}-cadence "
            f"frames at {render_h}x{render_w} "
            f"(fps={render_fps:.2f}, chunk_size={args.render_chunk_size})..."
        )
        color, depth = run_renderer_in_chunk_w_trj_mode(
            gaussians=gaussians,
            extrinsics=render_extr,
            intrinsics=render_intr,
            image_shape=(render_h, render_w),
            chunk_size=int(args.render_chunk_size),
            trj_mode="original",
            use_sh=True,
            color_mode="RGB+ED",
            enable_tqdm=True,
        )

        frames_render = (
            color[0].clamp(0.0, 1.0).mul(255.0).byte().permute(0, 2, 3, 1).cpu().numpy()
        )
        video_path = output_dir / "gs_trajectory.mp4"
        _save_video_mp4(str(video_path), frames_render, fps=render_fps)
        print(f"[vipe_da3_chunked_gs] Saved GS render video to {video_path}")

        del gaussians, render_extr, render_intr, color, depth, frames_render, final_gaussians
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    done_marker.write_text("done\n")
    print("[vipe_da3_chunked_gs] Done.")


if __name__ == "__main__":
    main()
