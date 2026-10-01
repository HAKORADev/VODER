# This file includes code originally from the Segment and Track Anything repository:
# https://github.com/z-x-yang/Segment-and-Track-Anything
# Licensed under the AGPL-3.0 License. See THIRD_PARTY_LICENSES.md for details.

from .fpn import FPNSegmentationHead


def build_decoder(name, **kwargs):
    if name == "fpn":
        return FPNSegmentationHead(**kwargs)
    else:
        raise NotImplementedError
