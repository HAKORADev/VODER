# This file includes code originally from the Segment and Track Anything repository:
# https://github.com/z-x-yang/Segment-and-Track-Anything
# Licensed under the AGPL-3.0 License. See THIRD_PARTY_LICENSES.md for details.

from torch import nn

from ..layers.normalization import FrozenBatchNorm2d
from .resnet import ResNet50


def build_encoder(name, frozen_bn=True, freeze_at=-1):
    if frozen_bn:
        BatchNorm = FrozenBatchNorm2d
    else:
        BatchNorm = nn.BatchNorm2d

    if name == "resnet50":
        return ResNet50(16, BatchNorm, freeze_at=freeze_at)
    else:
        raise NotImplementedError
