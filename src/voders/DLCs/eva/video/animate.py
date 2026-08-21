import os
import sys

_src_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

ANIMATE_MAX_DIMENSION = 1280
ANIMATE_SUPPORTED_RESOLUTIONS = ["832x480", "480x832", "1024x576", "576x1024", "1280x720", "720x1280"]
ANIMATE_DEFAULT_CLIP_LEN = 77
ANIMATE_DEFAULT_STEPS = 20

ENV_KEY = "animate"


class AnimateWrapper:
    def __init__(self):
        self.pipeline = None

    def ensure_model(self):
        from voders.DLCs.eva._envrunner import venv_exists
        if venv_exists(ENV_KEY):
            return True
        print(f"Wan2.2-Animate env not set up. Run: python setup.py --envs {ENV_KEY}")
        return False

    def animify(self, reference_image, pose_video, output_path, prompt="", seed=-1,
                clip_len=ANIMATE_DEFAULT_CLIP_LEN, sampling_steps=ANIMATE_DEFAULT_STEPS,
                guide_scale=1.0, refert_num=1, replace_flag=False):
        from voders.DLCs.eva._envrunner import run_in_venv
        from voders.DLCs.eva.media_download import resolve_input_path
        ref_resolved = resolve_input_path(reference_image, media_type='image')
        if ref_resolved is None:
            return False
        pose_resolved = resolve_input_path(pose_video, media_type='video')
        if pose_resolved is None:
            return False
        print(f"Generating animation with Wan2.2-Animate...")
        spec = {
            "action": "animify",
            "reference_image": ref_resolved,
            "pose_video": pose_resolved,
            "output_path": output_path,
            "prompt": prompt,
            "seed": seed,
            "clip_len": clip_len,
            "sampling_steps": sampling_steps,
            "guide_scale": guide_scale,
            "refert_num": refert_num,
            "replace_flag": replace_flag,
        }
        result = run_in_venv(ENV_KEY, spec)
        if result.get("success"):
            print(f"\n✓ Success! Output saved to: {result.get('output_path', output_path)}")
            return True
        print(f"Error: {result.get('error', 'unknown')}")
        return False

    def cleanup(self):
        pass
