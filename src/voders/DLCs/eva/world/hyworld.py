import os
import sys

_src_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

HYWORLD_DEFAULT_RESOLUTION = "512x512"
HYWORLD_MAX_REFS = 3

ENV_KEY = "hyworld"


class HYWorldWrapper:
    def __init__(self):
        self.pipeline = None

    def ensure_model(self):
        from voders.DLCs.eva._envrunner import venv_exists
        if venv_exists(ENV_KEY):
            return True
        print(f"HY-World env not set up. Run: python setup.py --envs {ENV_KEY}")
        return False

    def generate(self, prompt, output_path, resolution=None, seed=0, reference_paths=None):
        from voders.DLCs.eva._envrunner import run_in_venv
        from voders.DLCs.eva.media_download import resolve_input_path, check_reference_limit
        from voders.DLCs.eva.image.sam import sam_auto_mask_for_edit, sam_apply_mask_to_image
        reference_paths = check_reference_limit(reference_paths, HYWORLD_MAX_REFS, 'HY-World 2.0')
        resolved_refs = []
        if reference_paths:
            for ref_path in reference_paths:
                resolved = resolve_input_path(ref_path, media_type='image')
                if resolved:
                    if reference_paths.index(ref_path) == 0 and len(reference_paths) == 1:
                        mask = sam_auto_mask_for_edit(resolved)
                        if mask is not None:
                            masked_path = resolved.rsplit('.', 1)[0] + '_masked.png'
                            sam_apply_mask_to_image(resolved, mask, masked_path)
                            resolved = masked_path
                            print(f"SAM: auto-masked subject in reference image")
                    resolved_refs.append(resolved)
        print(f"Generating 3D world with HY-World 2.0...")
        spec = {
            "action": "generate",
            "prompt": prompt,
            "output_path": output_path,
            "resolution": resolution or HYWORLD_DEFAULT_RESOLUTION,
            "seed": seed,
            "reference_paths": resolved_refs,
        }
        result = run_in_venv(ENV_KEY, spec)
        if result.get("success"):
            print(f"\n✓ Success! Output saved to: {result.get('output_path', output_path)}")
            return True
        print(f"Error: {result.get('error', 'unknown')}")
        return False

    def cleanup(self):
        pass
