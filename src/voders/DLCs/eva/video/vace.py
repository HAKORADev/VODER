import os
import sys

_src_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

VACE_DEFAULT_RESOLUTION = "832x480"
VACE_DEFAULT_DURATION = 5
VACE_MAX_DURATION = 5
VACE_SUPPORTED_RESOLUTIONS = ["832x480", "480x832", "1024x576", "576x1024"]
VACE_MAX_DIMENSION = 1280
VACE_MAX_REFS = 4

ENV_KEY = "vace"


class VACEWrapper:
    def __init__(self):
        self.pipeline = None

    def ensure_model(self):
        from voders.DLCs.eva._envrunner import venv_exists
        if venv_exists(ENV_KEY):
            return True
        print(f"Wan VACE env not set up. Run: python setup.py --envs {ENV_KEY}")
        return False

    def edit(self, input_path, prompt, output_path, reference_paths=None, resolution=None, duration=VACE_DEFAULT_DURATION, seed=0):
        from voders.DLCs.eva._envrunner import run_in_venv
        from voders.DLCs.eva.media_download import resolve_input_path, check_reference_limit
        from voders.DLCs.eva.downscale import check_and_downscale_input
        from voders.DLCs.eva.image.sam import sam_auto_mask_for_edit, sam_apply_mask_to_image
        resolved = resolve_input_path(input_path, media_type='video')
        if resolved is None:
            return False
        input_path = resolved
        input_path = check_and_downscale_input(input_path, VACE_MAX_DIMENSION, VACE_MAX_DIMENSION)
        reference_paths = check_reference_limit(reference_paths, VACE_MAX_REFS, 'VACE')
        if reference_paths:
            resolved_refs = []
            for ref_path in reference_paths:
                resolved_ref = resolve_input_path(ref_path, media_type='image')
                if resolved_ref:
                    resolved_ref = check_and_downscale_input(resolved_ref, VACE_MAX_DIMENSION, VACE_MAX_DIMENSION)
                    resolved_refs.append(resolved_ref)
            reference_paths = resolved_refs if resolved_refs else None
        print(f"Editing video with Wan VACE...")
        spec = {
            "action": "edit",
            "input_path": input_path,
            "prompt": prompt,
            "output_path": output_path,
            "reference_paths": reference_paths,
            "resolution": resolution or VACE_DEFAULT_RESOLUTION,
            "duration": duration,
            "seed": seed,
        }
        result = run_in_venv(ENV_KEY, spec)
        if result.get("success"):
            print(f"\n✓ Success! Output saved to: {result.get('output_path', output_path)}")
            return True
        print(f"Error: {result.get('error', 'unknown')}")
        return False

    def cleanup(self):
        pass
