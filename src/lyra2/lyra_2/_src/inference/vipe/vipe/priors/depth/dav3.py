# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import numpy as np
import torch

try:
    from depth_anything_3.api import DepthAnything3
    from depth_anything_3.api import logger as dav3_logger
except ModuleNotFoundError:
    DepthAnything3 = dav3_logger = None

from vipe.utils.cameras import CameraType
from vipe.utils.misc import unpack_optional

from .base import DepthEstimationInput, DepthEstimationModel, DepthEstimationResult, DepthType


class DepthAnything3Model(DepthEstimationModel):
    """
    https://github.com/ByteDance-Seed/Depth-Anything-3
    """

    def __init__(self) -> None:
        super().__init__()
        if DepthAnything3 is None or dav3_logger is None:
            raise RuntimeError(
                "depth-anything-3 is not found. You can install it via `pip install --no-build-isolation -e .[dav3]`"
            )

        dav3_logger.level = 0  # Disable logging timing information
        self.model = DepthAnything3.from_pretrained("depth-anything/DA3METRIC-LARGE")
        self.model = self.model.cuda().eval()

    @property
    def depth_type(self) -> DepthType:
        """
        DAv3 model offers metric depth proportional to focal length
        See: https://github.com/ByteDance-Seed/Depth-Anything-3?tab=readme-ov-file#-faq
        """
        return DepthType.METRIC_DEPTH

    def estimate(self, src: DepthEstimationInput) -> DepthEstimationResult:
        rgb: torch.Tensor = unpack_optional(src.rgb)
        assert rgb.dtype == torch.float32, "Input image should be float32"

        assert src.camera_type == CameraType.PINHOLE, "DAv3 only supports pinhole cameras"
        focal_length: float = unpack_optional(src.intrinsics)[0].item()

        if rgb.dim() == 3:
            rgb, batch_dim = rgb[None], False
        else:
            batch_dim = True

        rgb_images = [(rgb[idx].cpu().numpy() * 255).astype(np.uint8) for idx in range(rgb.shape[0])]

        with torch.no_grad():
            dav3_inference_result = self.model.inference(
                rgb_images,
                process_res_method="upper_bound_resize",
                process_res=504,
            )

        # Compute focal internally in DAv3 model
        dav3_camera_focal = focal_length / max(rgb.shape[2], rgb.shape[1]) * 504
        # Normalize to correct scale, 300 is the magic/commonly used number chosen by the model authors.
        dav3_metric_depth = dav3_inference_result.depth * dav3_camera_focal / 300.0
        dav3_sky_mask = dav3_inference_result.sky

        # Mark sky region as invalid depth, so that inaccurate predictions are not used for further computation.
        dav3_metric_depth = dav3_metric_depth * (~dav3_sky_mask).astype(dav3_metric_depth.dtype)
        dav3_metric_depth = torch.from_numpy(dav3_metric_depth).cuda()[None]
        dav3_metric_depth = torch.nn.functional.interpolate(dav3_metric_depth, rgb.shape[1:3], mode="nearest")[:, 0]

        if not batch_dim:
            dav3_metric_depth = dav3_metric_depth.squeeze(0)

        return DepthEstimationResult(metric_depth=dav3_metric_depth)
