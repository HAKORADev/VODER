# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
import time
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.distributed as dist
from einops import rearrange

from lyra_2._ext.imaginaire.utils import distributed, log, misc
from lyra_2._ext.imaginaire.utils.callback import Callback
from lyra_2._ext.imaginaire.visualize.video import save_img_or_video


@torch.jit.script
def _fused_nan_to_num(gradients: list[torch.Tensor]) -> None:
    for gradient in gradients:
        torch.nan_to_num(gradient, nan=0.0, posinf=0.0, neginf=0.0, out=gradient)


def _clone_to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", copy=True)
    if isinstance(value, Mapping):
        return type(value)((key, _clone_to_cpu(item)) for key, item in value.items())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return type(value)(_clone_to_cpu(item) for item in value)
    return value


class Lyra2LowPrecisionCallback(Callback):
    """Match the source experiment's bf16 casting without corrupting camera geometry."""

    _KEEP_FP32 = {
        "camera_w2c",
        "depth",
        "intrinsics",
        "buffer_depths",
        "buffer_w2cs",
        "target_w2cs",
        "buffer_intrinsics",
        "target_intrinsics",
        "buffer_points",
        "buffer_masks",
    }

    def on_train_start(self, model, iteration: int = 0) -> None:
        del iteration
        self.precision = model.precision
        if self.precision not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError(f"Unsupported model precision: {self.precision}")

    def _cast(self, data: dict[str, Any]) -> None:
        for key, value in data.items():
            if key not in self._KEEP_FP32 and isinstance(value, torch.Tensor) and torch.is_floating_point(value):
                data[key] = value.to(dtype=self.precision)

    def on_training_step_start(self, model, data: dict[str, Any], iteration: int = 0) -> None:
        del model, iteration
        self._cast(data)

    def on_validation_step_start(self, model, data: dict[str, Any], iteration: int = 0) -> None:
        del model, iteration
        self._cast(data)


class LocalGradClipCallback(Callback):
    """The target experiment's finite-gradient cleanup and norm clipping, without WandB."""

    def __init__(self, clip_norm: float = 0.1, force_finite: bool = True):
        self.clip_norm = float(clip_norm)
        self.force_finite = bool(force_finite)
        self.last_total_norm: float | None = None

    def on_before_optimizer_step(
        self,
        model_ddp,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        grad_scaler: torch.amp.GradScaler,
        iteration: int = 0,
    ) -> None:
        del optimizer, scheduler, grad_scaler, iteration
        model = model_ddp.module if isinstance(model_ddp, distributed.DistributedDataParallel) else model_ddp
        if self.force_finite:
            gradients = []
            for parameter in model.parameters():
                if parameter.grad is not None:
                    gradients.append(parameter.grad)
            _fused_nan_to_num(gradients)
        total_norm = model.clip_grad_norm_(self.clip_norm)
        if isinstance(total_norm, torch.Tensor):
            self.last_total_norm = float(total_norm.detach().float().cpu())
        else:
            self.last_total_norm = float(total_norm)


class TensorBoardCallback(Callback):
    """Write scalar training telemetry to the local job directory."""

    def on_train_start(self, model, iteration: int = 0) -> None:
        del model
        self.writer = None
        self.step_started_at = time.perf_counter()
        if distributed.is_rank0():
            from torch.utils.tensorboard import SummaryWriter

            log_dir = os.path.join(self.config.job.path_local, "tensorboard")
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=log_dir, purge_step=iteration if iteration > 0 else None)
            log.info(f"TensorBoard logs: {log_dir}")

    def on_training_step_start(self, model, data: dict[str, Any], iteration: int = 0) -> None:
        del model, data, iteration
        self.step_started_at = time.perf_counter()

    @distributed.rank0_only
    def on_training_step_end(self, model, data_batch, output_batch, loss, iteration: int = 0) -> None:
        del model, data_batch
        if self.writer is None:
            return
        self.writer.add_scalar("train/loss", float(loss.detach().float().cpu()), iteration)
        for name, value in output_batch.items():
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                self.writer.add_scalar(f"train/{name}", float(value.detach().float().cpu()), iteration)
        self.writer.add_scalar("timer/iteration", time.perf_counter() - self.step_started_at, iteration)
        grad_clip = next(
            (cb for cb in self.trainer.callbacks._callbacks if isinstance(cb, LocalGradClipCallback)),
            None,
        )
        if grad_clip is not None and grad_clip.last_total_norm is not None:
            self.writer.add_scalar("optim/grad_norm", grad_clip.last_total_norm, iteration)
        if iteration % self.config.trainer.logging_iter == 0:
            log.info(f"Iteration {iteration}: loss={float(loss.detach().float().cpu()):.6f}")
            self.writer.flush()

    @distributed.rank0_only
    def on_before_optimizer_step(
        self,
        model_ddp,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        grad_scaler: torch.amp.GradScaler,
        iteration: int = 0,
    ) -> None:
        del model_ddp, optimizer
        if self.writer is not None:
            self.writer.add_scalar("optim/lr", scheduler.get_last_lr()[0], iteration)
            self.writer.add_scalar("optim/grad_scale", grad_scaler.get_scale(), iteration)

    @distributed.rank0_only
    def on_train_end(self, model, iteration: int = 0) -> None:
        del model, iteration
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()


class LocalVideoCallback(Callback):
    """Run the source experiment's sampling path and save an MP4 locally."""

    def __init__(
        self,
        every_n: int = 400,
        num_sampling_step: int = 35,
        guidance: float = 1.0,
        fps: int = 16,
        run_at_start: bool = False,
    ):
        self.every_n = int(every_n)
        self.num_sampling_step = int(num_sampling_step)
        self.guidance = float(guidance)
        self.fps = int(fps)
        self.run_at_start = bool(run_at_start)
        self.saved_data_batch: dict[str, Any] | None = None

    def on_train_start(self, model, iteration: int = 0) -> None:
        del model, iteration
        self.local_dir = os.path.join(self.config.job.path_local, "visualizations")
        if distributed.is_rank0():
            os.makedirs(self.local_dir, exist_ok=True)
            log.info(f"Local visualizations: {self.local_dir}")

    def _should_run(self, iteration_after_step: int) -> bool:
        if self.every_n <= 0:
            return False
        return (iteration_after_step == 1 and self.run_at_start) or iteration_after_step % self.every_n == 0

    def on_training_step_start(self, model, data: dict[str, Any], iteration: int = 0) -> None:
        del model
        if self._should_run(iteration + 1):
            self.saved_data_batch = _clone_to_cpu(data)

    @staticmethod
    def _as_rgb_videos(item: torch.Tensor) -> list[torch.Tensor]:
        if item.shape[1] == 3:
            return [item]
        if item.shape[1] > 0 and item.shape[1] % 3 == 0:
            return list(rearrange(item, "b (n c) t h w -> n b c t h w", c=3).unbind(0))
        repeats = (3 + item.shape[1] - 1) // item.shape[1]
        return [item.repeat(1, repeats, 1, 1, 1)[:, :3]]

    @torch.no_grad()
    def on_training_step_end(self, model, data_batch, output_batch, loss, iteration: int = 0) -> None:
        del data_batch, output_batch, loss
        if not self._should_run(iteration):
            return
        if self.saved_data_batch is None:
            raise RuntimeError("Visualization batch was not captured before the training step")

        sample_batch = misc.to(self.saved_data_batch, device="cuda")
        self.saved_data_batch = None
        generated = model.generate_samples_from_batch(
            sample_batch,
            guidance=self.guidance,
            state_shape=None,
            n_sample=1,
            num_steps=self.num_sampling_step,
            is_negative_prompt=False,
            return_condition_state=True,
        )
        if isinstance(generated, tuple):
            sample = generated[0]
            extras = [item for item in generated[1:] if isinstance(item, torch.Tensor)]
        else:
            sample = generated
            extras = []
        sample = model.decode(sample).float().cpu()

        videos: list[torch.Tensor] = []
        for item in [sample, *[extra.float().cpu() for extra in extras]]:
            if item.min() >= -1.0e-3 and item.max() <= 1.0 + 1.0e-3:
                item = item * 2.0 - 1.0
            if item.shape[2] < sample.shape[2]:
                padding = torch.zeros(
                    item.shape[0], item.shape[1], sample.shape[2] - item.shape[2], item.shape[3], item.shape[4]
                )
                item = torch.cat([padding, item], dim=2)
            videos.extend(self._as_rgb_videos(item))

        if distributed.is_rank0():
            grid = (1.0 + torch.stack(videos, dim=0).clamp(-1, 1)) / 2.0
            grid = rearrange(grid[:, :1], "n b c t h w -> c t (n h) (b w)")
            save_img_or_video(grid, os.path.join(self.local_dir, f"sample_iter_{iteration:09d}"), fps=self.fps)
        if dist.is_initialized():
            dist.barrier()
        del sample_batch, generated, sample, extras, videos
        torch.cuda.empty_cache()
