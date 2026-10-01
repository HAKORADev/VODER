import os
import sys

_src_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

QWEN_DEFAULT_RESOLUTION = "1024x1024"
QWEN_MAX_DIMENSION = 2752
QWEN_SUPPORTED_RESOLUTIONS = [
    "512x512", "768x768", "1024x1024", "1536x1536", "2048x2048",
    "1024x768", "768x1024", "1536x1024", "1024x1536",
    "1280x720", "720x1280", "1920x1080", "1080x1920",
    "2400x1792", "1792x2400", "2528x1696", "1696x2528", "2752x1536", "1536x2752",
]
QWEN_DEFAULT_STEPS = 40
QWEN_MAX_REFS = 3

ENV_KEY = "qwen-image2.1-uc"


class QwenImageUCWrapper:
    def __init__(self):
        self.pipeline = None

    def ensure_model(self):
        from voders.DLCs.eva._envrunner import venv_exists
        if venv_exists(ENV_KEY):
            return True
        print(f"Qwen-Image-2.1 UC env not set up. Run: python setup.py --envs {ENV_KEY}")
        return False

    def generate(self, prompt, output_path, resolution=None, seed=0, reference_paths=None, num_inference_steps=QWEN_DEFAULT_STEPS):
        from voders.DLCs.eva._envrunner import run_in_venv
        from voders.DLCs.eva.downscale import validate_resolution
        from voders.DLCs.eva.media_download import check_reference_limit, resolve_references
        resolution = validate_resolution(resolution, QWEN_SUPPORTED_RESOLUTIONS, QWEN_DEFAULT_RESOLUTION, QWEN_MAX_DIMENSION)
        reference_paths = check_reference_limit(reference_paths, QWEN_MAX_REFS, 'Qwen-Image-2.1 UC')
        if reference_paths:
            resolved_refs = resolve_references(reference_paths, default_media_type='image')
            reference_paths = resolved_refs if resolved_refs else None
        spec = {
            "action": "generate",
            "prompt": prompt,
            "output_path": output_path,
            "resolution": resolution,
            "seed": seed,
            "reference_paths": reference_paths,
            "num_inference_steps": num_inference_steps,
        }
        result = run_in_venv(ENV_KEY, spec)
        if result.get("success"):
            print(f"\n✓ Success! Output saved to: {result.get('output_path', output_path)}")
            return True
        print(f"Error: {result.get('error', 'unknown')}")
        return False

    def edit(self, input_path, prompt, output_path, reference_paths=None, resolution=None, seed=0, num_inference_steps=QWEN_DEFAULT_STEPS):
        from voders.DLCs.eva._envrunner import run_in_venv
        from voders.DLCs.eva.downscale import validate_resolution, check_and_downscale_input
        from voders.DLCs.eva.media_download import resolve_input_path, check_reference_limit, resolve_references
        resolved = resolve_input_path(input_path, media_type='image')
        if resolved is None:
            return False
        input_path = resolved
        input_path = check_and_downscale_input(input_path, QWEN_MAX_DIMENSION, QWEN_MAX_DIMENSION)
        reference_paths = check_reference_limit(reference_paths, QWEN_MAX_REFS, 'Qwen-Image-2.1 UC')
        if reference_paths:
            resolved_refs = resolve_references(reference_paths, default_media_type='image')
            resolved_refs = [check_and_downscale_input(ref, QWEN_MAX_DIMENSION, QWEN_MAX_DIMENSION) for ref in resolved_refs]
            reference_paths = resolved_refs if resolved_refs else None
        if resolution:
            resolution = validate_resolution(resolution, QWEN_SUPPORTED_RESOLUTIONS, None, QWEN_MAX_DIMENSION)
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

    def generate_nbg(self, prompt, output_path, resolution=None, seed=0, num_inference_steps=QWEN_DEFAULT_STEPS):
        from voders.DLCs.eva._envrunner import run_in_venv
        from voders.DLCs.eva.downscale import validate_resolution
        resolution = validate_resolution(resolution, QWEN_SUPPORTED_RESOLUTIONS, QWEN_DEFAULT_RESOLUTION, QWEN_MAX_DIMENSION)
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
