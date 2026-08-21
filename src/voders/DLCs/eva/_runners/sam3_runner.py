import os
import sys
import json

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

SPEC_PATH = os.environ.get("EVA_SPEC_PATH")
RESULT_PATH = os.environ.get("EVA_RESULT_PATH")


def write_result(success, output_path=None, error=None, extra=None):
    payload = {"success": bool(success), "output_path": output_path, "error": error}
    if extra:
        payload.update(extra)
    if RESULT_PATH:
        with open(RESULT_PATH, "w") as f:
            json.dump(payload, f)
    print(json.dumps(payload, indent=2))


def load_spec():
    if not SPEC_PATH or not os.path.exists(SPEC_PATH):
        return None
    with open(SPEC_PATH, "r") as f:
        return json.load(f)


def main():
    spec = load_spec()
    if spec is None:
        write_result(False, error="No spec provided")
        return 1
    action = spec.get("action")
    handlers = {
        "auto_mask": handle_auto_mask,
        "segment_image": handle_segment_image,
        "segment_video": handle_segment_video,
    }
    handler = handlers.get(action)
    if handler is None:
        write_result(False, error=f"Unknown action '{action}'. Available: {list(handlers.keys())}")
        return 1
    try:
        return handler(spec)
    except Exception as e:
        import traceback
        traceback.print_exc()
        write_result(False, error=f"Unhandled exception: {e}")
        return 1


def _ensure_model(model_dir):
    from huggingface_hub import snapshot_download
    if not os.path.exists(os.path.join(model_dir, "sam3_image_config.json")):
        print(f"Downloading SAM 3.1 to {model_dir}...")
        snapshot_download(
            repo_id="facebook/sam3.1",
            local_dir=model_dir,
            token=os.environ.get("HF_TOKEN"),
        )


def handle_auto_mask(spec):
    import torch
    import numpy as np
    from PIL import Image
    from voders.DLCs.eva._paths import SAM3_DIR
    _ensure_model(SAM3_DIR)
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    input_path = spec["input_path"]
    output_path = spec["output_path"]
    print("Loading SAM 3.1...")
    model = build_sam3_image_model()
    processor = Sam3Processor(model)
    image = Image.open(input_path).convert("RGB")
    state = processor.set_image(image)
    w, h = image.size
    center_prompt = (w // 2, h // 2)
    output = processor.set_point_prompt(state=state, points=[center_prompt], labels=[1])
    masks = output.get("masks") if isinstance(output, dict) else getattr(output, "masks", None)
    if masks is None or len(masks) == 0:
        write_result(False, error="SAM produced no masks")
        return 1
    mask = masks[0]
    if hasattr(mask, "cpu"):
        mask = mask.cpu().numpy()
    mask = (mask.astype(np.uint8) * 255)
    result = Image.fromarray(mask, mode="L")
    result.save(output_path)
    print(f"Mask saved: {output_path}")
    write_result(True, output_path=output_path)
    return 0


def handle_segment_image(spec):
    write_result(False, error="segment_image not implemented yet — use auto_mask")
    return 1


def handle_segment_video(spec):
    write_result(False, error="segment_video not implemented yet")
    return 1


if __name__ == "__main__":
    sys.exit(main())
