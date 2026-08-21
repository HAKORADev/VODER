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
    handlers = {"zero_shot_classify": handle_zero_shot_classify, "encode": handle_encode}
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


def _load_model():
    import torch
    from transformers import AutoModel, AutoProcessor
    ckpt = "google/siglip2-giant-opt-patch16-384"
    print("Loading SigLIP 2 giant...")
    model = AutoModel.from_pretrained(ckpt, device_map="auto").eval()
    processor = AutoProcessor.from_pretrained(ckpt)
    return model, processor


def handle_zero_shot_classify(spec):
    import torch
    from PIL import Image
    model, processor = _load_model()
    image_path = spec["image_path"]
    labels = spec["candidate_labels"]
    output_path = spec["output_path"]
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=[image], text=labels, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=-1).cpu().numpy().tolist()[0]
    payload = [{"score": float(probs[i]), "label": labels[i]} for i in range(len(labels))]
    payload.sort(key=lambda x: -x["score"])
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Classification saved: {output_path}")
    write_result(True, output_path=output_path)
    return 0


def handle_encode(spec):
    import torch
    from PIL import Image
    model, processor = _load_model()
    image_path = spec["image_path"]
    output_path = spec["output_path"]
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=[image], return_tensors="pt").to(model.device)
    with torch.no_grad():
        embeddings = model.get_image_features(**inputs)
    arr = embeddings.cpu().numpy()
    import numpy as np
    np.save(output_path, arr)
    print(f"Embeddings saved: {output_path}")
    write_result(True, output_path=output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
