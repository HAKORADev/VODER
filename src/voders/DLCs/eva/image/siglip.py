import os
import sys
import tempfile

_src_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

ENV_KEY = "siglip2"


def siglip_load():
    from voders.DLCs.eva._envrunner import venv_exists
    if venv_exists(ENV_KEY):
        return True
    print(f"SigLIP 2 env not set up. Run: python setup.py --envs {ENV_KEY}")
    return False


def siglip_encode_image(image_path):
    if not siglip_load():
        return None
    try:
        from voders.DLCs.eva._envrunner import run_in_venv
        tmp = tempfile.NamedTemporaryFile(suffix="_emb.npy", delete=False)
        output_path = tmp.name
        tmp.close()
        spec = {
            "action": "encode",
            "image_path": image_path,
            "output_path": output_path,
        }
        result = run_in_venv(ENV_KEY, spec)
        if not result.get("success"):
            print(f"SigLIP encode error: {result.get('error', 'unknown')}")
            return None
        import numpy as np
        arr = np.load(output_path)
        try:
            os.remove(output_path)
        except Exception:
            pass
        import torch
        return torch.from_numpy(arr)
    except Exception as e:
        print(f"SigLIP encode image error: {e}")
        return None


def siglip_encode_text(text):
    print("SigLIP 2 text encoding is not exposed via the venv runner yet (image-only).")
    return None


def siglip_zero_shot_classify(image_path, candidate_labels):
    if not siglip_load():
        return None
    try:
        from voders.DLCs.eva._envrunner import run_in_venv
        import tempfile
        import json
        tmp = tempfile.NamedTemporaryFile(suffix="_cls.json", delete=False)
        output_path = tmp.name
        tmp.close()
        spec = {
            "action": "zero_shot_classify",
            "image_path": image_path,
            "candidate_labels": candidate_labels,
            "output_path": output_path,
        }
        result = run_in_venv(ENV_KEY, spec)
        if not result.get("success"):
            print(f"SigLIP classify error: {result.get('error', 'unknown')}")
            return None
        with open(output_path, "r") as f:
            payload = json.load(f)
        try:
            os.remove(output_path)
        except Exception:
            pass
        return payload
    except Exception as e:
        print(f"SigLIP classify error: {e}")
        return None


def siglip_cleanup():
    pass
