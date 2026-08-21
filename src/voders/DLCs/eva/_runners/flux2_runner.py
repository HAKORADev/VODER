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
    if action is None:
        write_result(False, error="Spec missing 'action' field")
        return 1
    handlers = {
        "generate": handle_generate,
        "edit": handle_edit,
        "generate_nbg": handle_generate_nbg,
        "mini_gen": handle_mini_gen,
        "mini_edit": handle_mini_edit,
        "mini_nbg": handle_mini_nbg,
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


def _load_pipeline(model_dir):
    import torch
    from huggingface_hub import snapshot_download
    if not os.path.exists(os.path.join(model_dir, "model_index.json")):
        print(f"Downloading Flux 2 Dev to {model_dir} (~64GB)...")
        snapshot_download(
            repo_id="black-forest-labs/FLUX.2-dev",
            local_dir=model_dir,
            token=os.environ.get("HF_TOKEN"),
        )
    print("Loading Flux 2 Dev pipeline...")
    from diffusers import Flux2Pipeline
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    pipe = Flux2Pipeline.from_pretrained(
        model_dir,
        torch_dtype=dtype,
        token=os.environ.get("HF_TOKEN"),
    )
    pipe = pipe.to(device)
    print("Flux 2 Dev loaded.")
    return pipe, device, dtype


def _parse_resolution(resolution):
    if not resolution:
        return 1024, 1024
    try:
        parts = str(resolution).lower().split("x")
        w, h = int(parts[0]), int(parts[1])
        if max(w, h) > 2048:
            scale = 2048 / max(w, h)
            w, h = int(w * scale), int(h * scale)
        return w, h
    except Exception:
        return 1024, 1024


def handle_generate(spec):
    import torch
    from PIL import Image
    from voders.DLCs.eva._paths import FLUX2_DIR
    pipe, device, dtype = _load_pipeline(FLUX2_DIR)
    prompt = spec["prompt"]
    output_path = spec["output_path"]
    resolution = spec.get("resolution")
    seed = int(spec.get("seed", 0))
    steps = int(spec.get("num_inference_steps", 28))
    width, height = _parse_resolution(resolution)
    print(f"Generating image ({width}x{height}) with Flux 2 Dev...")
    generator = torch.Generator(device=device).manual_seed(seed)
    result = pipe(
        prompt=prompt,
        width=width,
        height=height,
        num_inference_steps=steps,
        generator=generator,
        guidance_scale=3.5,
    )
    image = result.images[0]
    image.save(output_path, format="PNG")
    print(f"Image generated: {output_path}")
    write_result(True, output_path=output_path)
    return 0


def handle_edit(spec):
    import torch
    from PIL import Image
    from voders.DLCs.eva._paths import FLUX2_DIR
    pipe, device, dtype = _load_pipeline(FLUX2_DIR)
    input_path = spec["input_path"]
    prompt = spec["prompt"]
    output_path = spec["output_path"]
    references = spec.get("reference_paths") or []
    resolution = spec.get("resolution")
    seed = int(spec.get("seed", 0))
    steps = int(spec.get("num_inference_steps", 28))
    input_image = Image.open(input_path).convert("RGB")
    if not resolution:
        w, h = input_image.size
        if max(w, h) > 2048:
            scale = 2048 / max(w, h)
            w, h = int(w * scale), int(h * scale)
        resolution = f"{w}x{h}"
    width, height = _parse_resolution(resolution)
    ref_images = None
    if references:
        ref_images = [Image.open(p).convert("RGB") for p in references if os.path.exists(p)]
        if not ref_images:
            ref_images = None
    print(f"Editing image ({width}x{height}) with Flux 2 Dev...")
    generator = torch.Generator(device=device).manual_seed(seed)
    kwargs = dict(
        prompt=prompt,
        image=input_image,
        width=width,
        height=height,
        num_inference_steps=steps,
        generator=generator,
        guidance_scale=3.5,
    )
    if ref_images:
        kwargs["reference_images"] = ref_images
    result = pipe(**kwargs)
    image = result.images[0]
    image.save(output_path, format="PNG")
    print(f"Image edited: {output_path}")
    write_result(True, output_path=output_path)
    return 0


def handle_generate_nbg(spec):
    import torch
    import numpy as np
    from PIL import Image
    from voders.DLCs.eva._paths import FLUX2_DIR
    pipe, device, dtype = _load_pipeline(FLUX2_DIR)
    prompt = spec["prompt"]
    output_path = spec["output_path"]
    resolution = spec.get("resolution")
    seed = int(spec.get("seed", 0))
    steps = int(spec.get("num_inference_steps", 28))
    nbg_prompt = (
        f"{prompt}. The subject is centered on a solid pure green (#00FF00) background "
        "with no other elements. The background is a flat, uniform green color with no "
        "texture, gradient, or variation. The subject is fully contained within the frame "
        "and does not touch the edges."
    )
    temp_path = output_path.replace(".png", "_greenbg.png")
    width, height = _parse_resolution(resolution)
    generator = torch.Generator(device=device).manual_seed(seed)
    result = pipe(
        prompt=nbg_prompt,
        width=width,
        height=height,
        num_inference_steps=steps,
        generator=generator,
        guidance_scale=3.5,
    )
    result.images[0].save(temp_path, format="PNG")
    print("Removing green background...")
    img = Image.open(temp_path).convert("RGBA")
    arr = np.array(img)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    green_mask = (g > 100) & (g > r + 30) & (g > b + 30)
    arr[green_mask, 3] = 0
    result_img = Image.fromarray(arr)
    result_img.save(output_path, format="PNG")
    if os.path.exists(temp_path):
        os.remove(temp_path)
    print(f"Transparent PNG saved: {output_path}")
    write_result(True, output_path=output_path)
    return 0


def _load_klein_pipeline(model_dir):
    import torch
    from huggingface_hub import snapshot_download
    if not os.path.exists(os.path.join(model_dir, "model_index.json")):
        print(f"Downloading Flux 2 Klein 9B to {model_dir} (~18GB)...")
        snapshot_download(
            repo_id="black-forest-labs/FLUX.2-klein-9B",
            local_dir=model_dir,
            token=os.environ.get("HF_TOKEN"),
        )
    print("Loading Flux 2 Klein 9B pipeline...")
    from diffusers import Flux2KleinPipeline
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    pipe = Flux2KleinPipeline.from_pretrained(
        model_dir,
        torch_dtype=dtype,
        token=os.environ.get("HF_TOKEN"),
    )
    pipe = pipe.to(device)
    print("Flux 2 Klein 9B loaded.")
    return pipe, device, dtype


KLEIN_DEFAULT_STEPS = 50
KLEIN_GUIDANCE_SCALE = 4.0
KLEIN_EDIT_STEPS = 50
KLEIN_EDIT_GUIDANCE_SCALE = 8.0
KLEIN_MAX_REFS = 4


def handle_mini_gen(spec):
    import torch
    from voders.DLCs.eva._paths import KLEIN_DIR
    pipe, device, dtype = _load_klein_pipeline(KLEIN_DIR)
    prompt = spec["prompt"]
    output_path = spec["output_path"]
    resolution = spec.get("resolution")
    seed = int(spec.get("seed", 0))
    steps = int(spec.get("num_inference_steps", KLEIN_DEFAULT_STEPS))
    width, height = _parse_resolution(resolution)
    print(f"Generating image ({width}x{height}) with Flux 2 Klein 9B...")
    generator = torch.Generator(device=device).manual_seed(seed)
    result = pipe(
        prompt=prompt,
        width=width,
        height=height,
        num_inference_steps=steps,
        generator=generator,
        guidance_scale=KLEIN_GUIDANCE_SCALE,
    )
    image = result.images[0]
    image.save(output_path, format="PNG")
    print(f"Image generated: {output_path}")
    write_result(True, output_path=output_path)
    return 0


def handle_mini_edit(spec):
    import torch
    from PIL import Image
    from voders.DLCs.eva._paths import KLEIN_DIR
    pipe, device, dtype = _load_klein_pipeline(KLEIN_DIR)
    input_path = spec["input_path"]
    prompt = spec["prompt"]
    output_path = spec["output_path"]
    references = spec.get("reference_paths") or []
    resolution = spec.get("resolution")
    seed = int(spec.get("seed", 0))
    steps = int(spec.get("num_inference_steps", KLEIN_EDIT_STEPS))
    input_image = Image.open(input_path).convert("RGB")
    if not resolution:
        w, h = input_image.size
        if max(w, h) > 2048:
            scale = 2048 / max(w, h)
            w, h = int(w * scale), int(h * scale)
        resolution = f"{w}x{h}"
    width, height = _parse_resolution(resolution)
    ref_images = None
    if references:
        ref_images = [Image.open(p).convert("RGB") for p in references if os.path.exists(p)]
        if not ref_images:
            ref_images = None
    print(f"Editing image ({width}x{height}) with Flux 2 Klein 9B...")
    generator = torch.Generator(device=device).manual_seed(seed)
    from diffusers import Flux2KleinInpaintPipeline
    inpaint_pipe = Flux2KleinInpaintPipeline.from_pipe(pipe)
    kwargs = dict(
        prompt=prompt,
        image=input_image,
        width=width,
        height=height,
        num_inference_steps=steps,
        generator=generator,
        guidance_scale=KLEIN_EDIT_GUIDANCE_SCALE,
    )
    if ref_images:
        kwargs["image_reference"] = ref_images[0]
    result = inpaint_pipe(**kwargs)
    image = result.images[0]
    image.save(output_path, format="PNG")
    print(f"Image edited: {output_path}")
    write_result(True, output_path=output_path)
    return 0


def handle_mini_nbg(spec):
    import torch
    import numpy as np
    from PIL import Image
    from voders.DLCs.eva._paths import KLEIN_DIR
    pipe, device, dtype = _load_klein_pipeline(KLEIN_DIR)
    prompt = spec["prompt"]
    output_path = spec["output_path"]
    resolution = spec.get("resolution")
    seed = int(spec.get("seed", 0))
    steps = int(spec.get("num_inference_steps", KLEIN_DEFAULT_STEPS))
    nbg_prompt = (
        f"{prompt}. The subject is centered on a solid pure green (#00FF00) background "
        "with no other elements. The background is a flat, uniform green color with no "
        "texture, gradient, or variation. The subject is fully contained within the frame "
        "and does not touch the edges."
    )
    temp_path = output_path.replace(".png", "_greenbg.png")
    width, height = _parse_resolution(resolution)
    generator = torch.Generator(device=device).manual_seed(seed)
    result = pipe(
        prompt=nbg_prompt,
        width=width,
        height=height,
        num_inference_steps=steps,
        generator=generator,
        guidance_scale=KLEIN_GUIDANCE_SCALE,
    )
    result.images[0].save(temp_path, format="PNG")
    print("Removing green background...")
    img = Image.open(temp_path).convert("RGBA")
    arr = np.array(img)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    green_mask = (g > 100) & (g > r + 30) & (g > b + 30)
    arr[green_mask, 3] = 0
    result_img = Image.fromarray(arr)
    result_img.save(output_path, format="PNG")
    if os.path.exists(temp_path):
        os.remove(temp_path)
    print(f"Transparent PNG saved: {output_path}")
    write_result(True, output_path=output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
