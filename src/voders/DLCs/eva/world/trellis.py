import os
import sys

_src_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

ENV_KEY = "trellis"


class Trellis2Wrapper:
    def __init__(self):
        self.pipeline = None

    def ensure_model(self):
        from voders.DLCs.eva._envrunner import venv_exists
        if venv_exists(ENV_KEY):
            return True
        print(f"TRELLIS.2 env not set up. Run: python setup.py --envs {ENV_KEY}")
        return False

    def objectify(self, input_path, output_path, seed=0):
        from voders.DLCs.eva._envrunner import run_in_venv
        from voders.DLCs.eva.media_download import resolve_input_path
        resolved = resolve_input_path(input_path, media_type='image')
        if resolved is None:
            return False
        input_path = resolved
        print(f"Converting image to 3D object with TRELLIS.2...")
        spec = {
            "action": "objectify",
            "input_path": input_path,
            "output_path": output_path,
            "seed": seed,
        }
        result = run_in_venv(ENV_KEY, spec)
        if result.get("success"):
            print(f"\n✓ Success! 3D object saved to: {result.get('output_path', output_path)}")
            return True
        print(f"Error: {result.get('error', 'unknown')}")
        return False

    def edit(self, input_path, reference_image, output_path, seed=0):
        from voders.DLCs.eva._envrunner import run_in_venv
        from voders.DLCs.eva.media_download import resolve_input_path
        resolved_input = resolve_input_path(input_path, media_type='image')
        if resolved_input is None:
            if os.path.exists(input_path):
                resolved_input = input_path
            else:
                print(f"Error: input mesh not found: {input_path}")
                return False
        resolved_ref = resolve_input_path(reference_image, media_type='image')
        if resolved_ref is None:
            return False
        print(f"Retexturing 3D object with TRELLIS.2...")
        spec = {
            "action": "edit",
            "input_path": resolved_input,
            "reference_image": resolved_ref,
            "output_path": output_path,
            "seed": seed,
        }
        result = run_in_venv(ENV_KEY, spec)
        if result.get("success"):
            print(f"\n✓ Success! Retextured object saved to: {result.get('output_path', output_path)}")
            return True
        print(f"Error: {result.get('error', 'unknown')}")
        return False

    def cleanup(self):
        pass
