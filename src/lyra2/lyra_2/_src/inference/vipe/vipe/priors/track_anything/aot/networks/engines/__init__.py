# This file includes code originally from the Segment and Track Anything repository:
# https://github.com/z-x-yang/Segment-and-Track-Anything
# Licensed under the AGPL-3.0 License. See THIRD_PARTY_LICENSES.md for details.

from ..engines.aot_engine import AOTEngine, AOTInferEngine
from ..engines.deaot_engine import DeAOTEngine, DeAOTInferEngine


def build_engine(name, phase="train", **kwargs):
    if name == "aotengine":
        if phase == "train":
            return AOTEngine(**kwargs)
        elif phase == "eval":
            return AOTInferEngine(**kwargs)
        else:
            raise NotImplementedError
    elif name == "deaotengine":
        if phase == "train":
            return DeAOTEngine(**kwargs)
        elif phase == "eval":
            return DeAOTInferEngine(**kwargs)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
