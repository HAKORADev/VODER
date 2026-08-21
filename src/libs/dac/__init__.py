__version__ = "1.0.0"

__model_version__ = "latest"

try:
    import audiotools
    audiotools.ml.BaseModel.INTERN += ["dac.**"]
    audiotools.ml.BaseModel.EXTERN += ["einops"]
except ImportError:
    pass

from . import nn
from . import model
