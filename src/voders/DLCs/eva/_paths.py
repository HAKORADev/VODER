import os

_EVA_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.dirname(_EVA_DIR)))
CHECKPOINTS_DIR = os.path.join(_SRC_DIR, "models", "checkpoints")

FLUX2_DIR = os.path.join(CHECKPOINTS_DIR, "flux2_dev")
KLEIN_DIR = os.path.join(CHECKPOINTS_DIR, "flux2_klein_9b")
H3_DIR = os.path.join(CHECKPOINTS_DIR, "minimax_h3")
VACE_DIR = os.path.join(CHECKPOINTS_DIR, "wan_vace_14b")
ANIMATE_DIR = os.path.join(CHECKPOINTS_DIR, "wan2_2_animate_14b")
S2V_DIR = os.path.join(CHECKPOINTS_DIR, "wan2_2_s2v_14b")
HYWORLD_DIR = os.path.join(CHECKPOINTS_DIR, "hy_world")
TRELLIS_DIR = os.path.join(CHECKPOINTS_DIR, "trellis2")
SAM3_DIR = os.path.join(CHECKPOINTS_DIR, "sam3")
SIGLIP2_DIR = os.path.join(CHECKPOINTS_DIR, "siglip2_giant")

for _d in (FLUX2_DIR, KLEIN_DIR, H3_DIR, VACE_DIR, ANIMATE_DIR, S2V_DIR, HYWORLD_DIR, TRELLIS_DIR, SAM3_DIR, SIGLIP2_DIR):
    os.makedirs(_d, exist_ok=True)
