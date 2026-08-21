import os
import sys
import tempfile

_src_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

ENV_KEY = "sam3"


def sam_load_model():
    from voders.DLCs.eva._envrunner import venv_exists
    if venv_exists(ENV_KEY):
        return True
    print(f"SAM 3.1 env not set up. Run: python setup.py --envs {ENV_KEY}")
    return False


def sam_auto_mask_for_edit(image_path, output_mask_path=None, include_subject=True):
    if not sam_load_model():
        return None
    try:
        from voders.DLCs.eva._envrunner import run_in_venv
        if output_mask_path is None:
            tmp = tempfile.NamedTemporaryFile(suffix="_mask.png", delete=False)
            output_mask_path = tmp.name
            tmp.close()
        spec = {
            "action": "auto_mask",
            "input_path": image_path,
            "output_path": output_mask_path,
        }
        result = run_in_venv(ENV_KEY, spec)
        if not result.get("success"):
            print(f"SAM auto-mask error: {result.get('error', 'unknown')}")
            return None
        import numpy as np
        from PIL import Image
        mask_img = Image.open(output_mask_path).convert("L")
        mask = np.array(mask_img) > 127
        if not include_subject:
            mask = ~mask
        return mask
    except Exception as e:
        print(f"SAM auto-mask error: {e}")
        return None


def sam_apply_mask_to_image(image_path, mask, output_path, invert=False):
    try:
        from PIL import Image
        import numpy as np
        img = Image.open(image_path).convert("RGBA")
        arr = np.array(img)
        if invert:
            mask = ~mask
        arr[~mask, 3] = 0
        result = Image.fromarray(arr)
        result.save(output_path)
        return output_path
    except Exception as e:
        print(f"SAM apply mask error: {e}")
        return image_path


def sam_segment_image(image_path, points=None, boxes=None, output_mask_path=None):
    if not sam_load_model():
        return None
    if output_mask_path is None:
        tmp = tempfile.NamedTemporaryFile(suffix="_mask.png", delete=False)
        output_mask_path = tmp.name
        tmp.close()
    try:
        from voders.DLCs.eva._envrunner import run_in_venv
        spec = {
            "action": "segment_image",
            "input_path": image_path,
            "output_path": output_mask_path,
            "points": points,
            "boxes": boxes,
        }
        result = run_in_venv(ENV_KEY, spec)
        if not result.get("success"):
            print(f"SAM segment_image error: {result.get('error', 'unknown')}")
            return None
        import numpy as np
        from PIL import Image
        mask_img = Image.open(output_mask_path).convert("L")
        mask = np.array(mask_img) > 127
        return mask
    except Exception as e:
        print(f"SAM segmentation error: {e}")
        return None


def sam_segment_video(video_path, points=None, output_mask_dir=None):
    if not sam_load_model():
        return None
    try:
        from voders.DLCs.eva._envrunner import run_in_venv
        spec = {
            "action": "segment_video",
            "input_path": video_path,
            "output_dir": output_mask_dir,
            "points": points,
        }
        result = run_in_venv(ENV_KEY, spec)
        if not result.get("success"):
            print(f"SAM segment_video error: {result.get('error', 'unknown')}")
            return None
        return result.get("masks", [])
    except Exception as e:
        print(f"SAM video segmentation error: {e}")
        return None


def sam_cleanup():
    pass
