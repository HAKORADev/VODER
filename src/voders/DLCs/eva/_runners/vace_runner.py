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
    handlers = {"edit": handle_edit, "generate": handle_generate}
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


def _load_pipeline(model_dir):
    import torch
    from huggingface_hub import snapshot_download
    if not os.path.exists(os.path.join(model_dir, "config.json")):
        print(f"Downloading Wan 2.1 VACE 14B to {model_dir} (large)...")
        snapshot_download(
            repo_id="Wan-AI/Wan2.1-VACE-14B",
            local_dir=model_dir,
            token=os.environ.get("HF_TOKEN"),
        )
    print("Loading Wan 2.1 VACE 14B pipeline...")
    from wan21.vace import VACEPipeline
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    pipe = VACEPipeline.from_pretrained(model_dir, torch_dtype=dtype)
    pipe = pipe.to(device)
    print("Wan 2.1 VACE 14B loaded.")
    return pipe, device


def handle_generate(spec):
    import torch
    from voders.DLCs.eva._paths import VACE_DIR
    pipe, device = _load_pipeline(VACE_DIR)
    prompt = spec["prompt"]
    output_path = spec["output_path"]
    duration = int(spec.get("duration", 5))
    resolution = spec.get("resolution", "832x480")
    seed = int(spec.get("seed", 0))
    image_refs = spec.get("image_refs") or []
    video_refs = spec.get("video_refs") or []
    generator = torch.Generator(device=device).manual_seed(seed)
    try:
        parts = resolution.lower().split("x")
        width, height = int(parts[0]), int(parts[1])
    except Exception:
        width, height = 832, 480
    num_frames = max(1, int(duration * 24))
    print(f"Generating video with Wan VACE ({width}x{height}, {num_frames} frames)...")
    kwargs = dict(
        prompt=prompt,
        negative_prompt="",
        num_frames=num_frames,
        width=width,
        height=height,
        num_inference_steps=40,
        guidance_scale=7.5,
        generator=generator,
    )
    if image_refs:
        from PIL import Image
        kwargs["images"] = [Image.open(p).convert("RGB") for p in image_refs]
    if video_refs:
        kwargs["videos"] = video_refs
    output = pipe(**kwargs)
    if hasattr(output, "videos"):
        video = output.videos[0]
    elif isinstance(output, dict):
        video = output.get("video", output.get("videos", None))
    else:
        video = output
    if video is None:
        write_result(False, error="VACE produced no output")
        return 1
    import imageio
    import numpy as np
    if isinstance(video, torch.Tensor):
        video = video.cpu().numpy()
    if isinstance(video, list):
        video = video[0] if video else None
    if isinstance(video, np.ndarray):
        if video.ndim == 4:
            video = video[0]
        if video.dtype != np.uint8:
            video = ((video + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
        if video.shape[-1] in (3, 4):
            video = video.transpose(0, 2, 3, 1)
        elif video.shape[0] in (3, 4):
            video = video.transpose(1, 2, 0)
    imageio.mimsave(output_path, video, fps=24)
    print(f"Video generated: {output_path}")
    write_result(True, output_path=output_path)
    return 0


def handle_edit(spec):
    import torch
    from voders.DLCs.eva._paths import VACE_DIR
    pipe, device = _load_pipeline(VACE_DIR)
    input_path = spec["input_path"]
    prompt = spec["prompt"]
    output_path = spec["output_path"]
    references = spec.get("reference_paths") or []
    duration = int(spec.get("duration", 5))
    seed = int(spec.get("seed", 0))
    generator = torch.Generator(device=device).manual_seed(seed)
    num_frames = max(1, int(duration * 24))
    print(f"Editing video with Wan VACE...")
    from PIL import Image
    kwargs = dict(
        prompt=prompt,
        negative_prompt="",
        video=input_path,
        num_frames=num_frames,
        num_inference_steps=40,
        guidance_scale=7.5,
        generator=generator,
    )
    if references:
        kwargs["images"] = [Image.open(p).convert("RGB") for p in references if os.path.exists(p)]
    output = pipe(**kwargs)
    if hasattr(output, "videos"):
        video = output.videos[0]
    elif isinstance(output, dict):
        video = output.get("video", output.get("videos", None))
    else:
        video = output
    if video is None:
        write_result(False, error="VACE produced no output")
        return 1
    import imageio
    import numpy as np
    if isinstance(video, torch.Tensor):
        video = video.cpu().numpy()
    if isinstance(video, list):
        video = video[0] if video else None
    if isinstance(video, np.ndarray):
        if video.ndim == 4:
            video = video[0]
        if video.dtype != np.uint8:
            video = ((video + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
        if video.shape[-1] in (3, 4):
            video = video.transpose(0, 2, 3, 1)
        elif video.shape[0] in (3, 4):
            video = video.transpose(1, 2, 0)
    imageio.mimsave(output_path, video, fps=24)
    print(f"Video edited: {output_path}")
    write_result(True, output_path=output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
