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

from enum import Enum


class DataField(Enum):
    """Fields stored by the local DL3DV/Radym dataset used for Lyra 2 training."""

    IMAGE_RGB = "image_rgb"
    CAMERA_C2W_TRANSFORM = "camera_c2w_transform"
    CAMERA_INTRINSICS = "camera_intrinsics"
    CAPTION = "caption"
    METRIC_DEPTH = "metric_depth"
    DYNAMIC_INSTANCE_MASK = "dynamic_instance_mask"
    BACKWARD_FLOW = "backward_flow"
    OBJECT_BBOX = "object_bbox"
