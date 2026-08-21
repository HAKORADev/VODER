import os
import sys

_src_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

FLUX2_DEFAULT_RESOLUTION = "1024x1024"
FLUX2_MAX_DIMENSION = 2048
FLUX2_SUPPORTED_RESOLUTIONS = [
    "512x512", "768x768", "1024x1024", "1536x1536", "2048x2048",
    "1024x768", "768x1024", "1536x1024", "1024x1536",
    "1920x1080", "1080x1920", "1280x720", "720x1280",
]
FLUX2_DEFAULT_STEPS = 28
FLUX2_GUIDANCE_SCALE = 3.5
FLUX2_MAX_REFS = 4

ENV_KEY = "flux2"


class Flux2Wrapper:
    def __init__(self):
        self.pipeline = None

    def ensure_model(self):
        from voders.DLCs.eva._envrunner import venv_exists
        if venv_exists(ENV_KEY):
            return True
        print(f"Flux 2 Dev env not set up. Run: python setup.py --envs {ENV_KEY}")
        return False

    def generate(self, prompt, output_path, resolution=None, seed=0, num_inference_steps=FLUX2_DEFAULT_STEPS):
        from voders.DLCs.eva._envrunner import run_in_venv
        from voders.DLCs.eva.downscale import validate_resolution
        resolution = validate_resolution(resolution, FLUX2_SUPPORTED_RESOLUTIONS, FLUX2_DEFAULT_RESOLUTION, FLUX2_MAX_DIMENSION)
        spec = {
            "action": "generate",
            "prompt": prompt,
            "output_path": output_path,
            "resolution": resolution,
            "seed": seed,
            "num_inference_steps": num_inference_steps,
        }
        result = run_in_venv(ENV_KEY, spec)
        if result.get("success"):
            print(f"\n✓ Success! Output saved to: {result.get('output_path', output_path)}")
            return True
        print(f"Error: {result.get('error', 'unknown')}")
        return False

    def edit(self, input_path, prompt, output_path, reference_paths=None, resolution=None, seed=0, num_inference_steps=FLUX2_DEFAULT_STEPS):
        from voders.DLCs.eva._envrunner import run_in_venv
        from voders.DLCs.eva.downscale import validate_resolution
        from voders.DLCs.eva.media_download import resolve_input_path, check_reference_limit
        from voders.DLCs.eva.image.sam import sam_auto_mask_for_edit, sam_apply_mask_to_image
        resolved = resolve_input_path(input_path, media_type='image')
        if resolved is None:
            return False
        input_path = resolved
        from voders.DLCs.eva.downscale import check_and_downscale_input
        input_path = check_and_downscale_input(input_path, FLUX2_MAX_DIMENSION, FLUX2_MAX_DIMENSION)
        mask = sam_auto_mask_for_edit(input_path)
        if mask is not None:
            masked_path = input_path.rsplit('.', 1)[0] + '_masked.png'
            sam_apply_mask_to_image(input_path, mask, masked_path)
            input_path = masked_path
            print(f"SAM: auto-masked subject in input image")
        reference_paths = check_reference_limit(reference_paths, FLUX2_MAX_REFS, 'Flux 2 Dev')
        if reference_paths:
            resolved_refs = []
            for ref_path in reference_paths:
                resolved_ref = resolve_input_path(ref_path, media_type='image')
                if resolved_ref:
                    resolved_ref = check_and_downscale_input(resolved_ref, FLUX2_MAX_DIMENSION, FLUX2_MAX_DIMENSION)
                    resolved_refs.append(resolved_ref)
            reference_paths = resolved_refs if resolved_refs else None
        if resolution:
            resolution = validate_resolution(resolution, FLUX2_SUPPORTED_RESOLUTIONS, None, FLUX2_MAX_DIMENSION)
        spec = {
            "action": "edit",
            "input_path": input_path,
            "prompt": prompt,
            "output_path": output_path,
            "reference_paths": reference_paths,
            "resolution": resolution,
            "seed": seed,
            "num_inference_steps": num_inference_steps,
        }
        result = run_in_venv(ENV_KEY, spec)
        if result.get("success"):
            print(f"\n✓ Success! Output saved to: {result.get('output_path', output_path)}")
            return True
        print(f"Error: {result.get('error', 'unknown')}")
        return False

    def generate_nbg(self, prompt, output_path, resolution=None, seed=0, num_inference_steps=FLUX2_DEFAULT_STEPS):
        from voders.DLCs.eva._envrunner import run_in_venv
        from voders.DLCs.eva.downscale import validate_resolution
        resolution = validate_resolution(resolution, FLUX2_SUPPORTED_RESOLUTIONS, FLUX2_DEFAULT_RESOLUTION, FLUX2_MAX_DIMENSION)
        spec = {
            "action": "generate_nbg",
            "prompt": prompt,
            "output_path": output_path,
            "resolution": resolution,
            "seed": seed,
            "num_inference_steps": num_inference_steps,
        }
        result = run_in_venv(ENV_KEY, spec)
        if result.get("success"):
            print(f"\n✓ Success! Transparent PNG saved to: {result.get('output_path', output_path)}")
            return True
        print(f"Error: {result.get('error', 'unknown')}")
        return False

    def cleanup(self):
        pass


KLEIN_DEFAULT_RESOLUTION = "1024x1024"
KLEIN_MAX_DIMENSION = 2048
KLEIN_SUPPORTED_RESOLUTIONS = FLUX2_SUPPORTED_RESOLUTIONS
KLEIN_DEFAULT_STEPS = 50
KLEIN_GUIDANCE_SCALE = 4.0
KLEIN_EDIT_STEPS = 50
KLEIN_EDIT_GUIDANCE_SCALE = 8.0
KLEIN_MAX_REFS = 4


class KleinWrapper:
    def __init__(self):
        self.pipeline = None

    def ensure_model(self):
        from voders.DLCs.eva._envrunner import venv_exists
        if venv_exists(ENV_KEY):
            return True
        print(f"Flux 2 Klein env not set up. Run: python setup.py --envs {ENV_KEY}")
        return False

    def mini_gen(self, prompt, output_path, resolution=None, seed=0, num_inference_steps=KLEIN_DEFAULT_STEPS):
        from voders.DLCs.eva._envrunner import run_in_venv
        from voders.DLCs.eva.downscale import validate_resolution
        resolution = validate_resolution(resolution, KLEIN_SUPPORTED_RESOLUTIONS, KLEIN_DEFAULT_RESOLUTION, KLEIN_MAX_DIMENSION)
        spec = {
            "action": "mini_gen",
            "prompt": prompt,
            "output_path": output_path,
            "resolution": resolution,
            "seed": seed,
            "num_inference_steps": num_inference_steps,
        }
        result = run_in_venv(ENV_KEY, spec)
        if result.get("success"):
            print(f"\n✓ Success! Output saved to: {result.get('output_path', output_path)}")
            return True
        print(f"Error: {result.get('error', 'unknown')}")
        return False

    def mini_edit(self, input_path, prompt, output_path, reference_paths=None, resolution=None, seed=0, num_inference_steps=KLEIN_EDIT_STEPS):
        from voders.DLCs.eva._envrunner import run_in_venv
        from voders.DLCs.eva.downscale import validate_resolution, check_and_downscale_input
        from voders.DLCs.eva.media_download import resolve_input_path, check_reference_limit
        from voders.DLCs.eva.image.sam import sam_auto_mask_for_edit, sam_apply_mask_to_image
        resolved = resolve_input_path(input_path, media_type='image')
        if resolved is None:
            return False
        input_path = resolved
        input_path = check_and_downscale_input(input_path, KLEIN_MAX_DIMENSION, KLEIN_MAX_DIMENSION)
        mask = sam_auto_mask_for_edit(input_path)
        if mask is not None:
            masked_path = input_path.rsplit('.', 1)[0] + '_masked.png'
            sam_apply_mask_to_image(input_path, mask, masked_path)
            input_path = masked_path
            print(f"SAM: auto-masked subject in input image")
        reference_paths = check_reference_limit(reference_paths, KLEIN_MAX_REFS, 'Flux 2 Klein 9B')
        if reference_paths:
            resolved_refs = []
            for ref_path in reference_paths:
                resolved_ref = resolve_input_path(ref_path, media_type='image')
                if resolved_ref:
                    resolved_ref = check_and_downscale_input(resolved_ref, KLEIN_MAX_DIMENSION, KLEIN_MAX_DIMENSION)
                    resolved_refs.append(resolved_ref)
            reference_paths = resolved_refs if resolved_refs else None
        if resolution:
            resolution = validate_resolution(resolution, KLEIN_SUPPORTED_RESOLUTIONS, None, KLEIN_MAX_DIMENSION)
        spec = {
            "action": "mini_edit",
            "input_path": input_path,
            "prompt": prompt,
            "output_path": output_path,
            "reference_paths": reference_paths,
            "resolution": resolution,
            "seed": seed,
            "num_inference_steps": num_inference_steps,
        }
        result = run_in_venv(ENV_KEY, spec)
        if result.get("success"):
            print(f"\n✓ Success! Output saved to: {result.get('output_path', output_path)}")
            return True
        print(f"Error: {result.get('error', 'unknown')}")
        return False

    def mini_nbg(self, prompt, output_path, resolution=None, seed=0, num_inference_steps=KLEIN_DEFAULT_STEPS):
        from voders.DLCs.eva._envrunner import run_in_venv
        from voders.DLCs.eva.downscale import validate_resolution
        resolution = validate_resolution(resolution, KLEIN_SUPPORTED_RESOLUTIONS, KLEIN_DEFAULT_RESOLUTION, KLEIN_MAX_DIMENSION)
        spec = {
            "action": "mini_nbg",
            "prompt": prompt,
            "output_path": output_path,
            "resolution": resolution,
            "seed": seed,
            "num_inference_steps": num_inference_steps,
        }
        result = run_in_venv(ENV_KEY, spec)
        if result.get("success"):
            print(f"\n✓ Success! Transparent PNG saved to: {result.get('output_path', output_path)}")
            return True
        print(f"Error: {result.get('error', 'unknown')}")
        return False

    def cleanup(self):
        pass
