import os
import sys

_src_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

S2V_MAX_DIMENSION = 1280
S2V_SUPPORTED_RESOLUTIONS = ["832x480", "480x832", "1024x576", "576x1024", "1280x720", "720x1280"]
S2V_DEFAULT_FRAMES = 80
S2V_DEFAULT_STEPS = 40

ENV_KEY = "s2v"


class S2VWrapper:
    def __init__(self):
        self.pipeline = None

    def ensure_model(self):
        from voders.DLCs.eva._envrunner import venv_exists
        if venv_exists(ENV_KEY):
            return True
        print(f"Wan2.2-S2V env not set up. Run: python setup.py --envs {ENV_KEY}")
        return False

    def lipsync(self, reference_image, audio_path, output_path, prompt="", seed=-1,
                infer_frames=S2V_DEFAULT_FRAMES, sampling_steps=S2V_DEFAULT_STEPS,
                guide_scale=4.5, init_first_frame=True):
        from voders.DLCs.eva._envrunner import run_in_venv
        from voders.DLCs.eva.media_download import resolve_input_path
        ref_resolved = resolve_input_path(reference_image, media_type='image')
        if ref_resolved is None:
            return False
        if audio_path and os.path.exists(audio_path):
            audio_resolved = audio_path
        elif audio_path and (audio_path.startswith('http://') or audio_path.startswith('https://')):
            print(f"Error: audio URL download not supported for lipsync — download locally first: {audio_path}")
            return False
        else:
            print(f"Error: audio file not found: {audio_path}")
            return False
        print(f"Generating lip-sync video with Wan2.2-S2V...")
        spec = {
            "action": "lipsync",
            "reference_image": ref_resolved,
            "audio_path": audio_resolved,
            "output_path": output_path,
            "prompt": prompt,
            "seed": seed,
            "infer_frames": infer_frames,
            "sampling_steps": sampling_steps,
            "guide_scale": guide_scale,
            "init_first_frame": init_first_frame,
        }
        result = run_in_venv(ENV_KEY, spec)
        if result.get("success"):
            print(f"\n✓ Success! Output saved to: {result.get('output_path', output_path)}")
            return True
        print(f"Error: {result.get('error', 'unknown')}")
        return False

    def cleanup(self):
        pass
