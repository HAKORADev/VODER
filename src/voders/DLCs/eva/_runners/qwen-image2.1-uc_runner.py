import os
import sys
import json
import math
import shutil
import socket
import subprocess
import time
import urllib.request
import urllib.error

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

SPEC_PATH = os.environ.get("EVA_SPEC_PATH")
RESULT_PATH = os.environ.get("EVA_RESULT_PATH")

QWEN_REPO_ID = "0xSojalSec/Qwen-Image-2.1-Uncensored-HF"
QWEN_GGUF_FILE = "qwen-image-2.1-UC-Q4_K_M.gguf"
QWEN_CLIP_FILE = "text_encoders/qwen3vl_8b_int8_convrot.safetensors"
QWEN_VAE_FILE = "vae/qwen_image_2.1_vae_bf16.safetensors"
QWEN_GGUF_NAME = "qwen-image-2.1-UC-Q4_K_M.gguf"
QWEN_CLIP_NAME = "qwen3vl_8b_int8_convrot.safetensors"
QWEN_VAE_NAME = "qwen_image_2.1_vae_bf16.safetensors"
COMFYUI_REPO = "https://github.com/comfyanonymous/ComfyUI"
GGUF_NODE_REPO = "https://github.com/leejet/ComfyUI-GGUF"
QWEN_STEPS = 40
QWEN_CFG = 1.0
QWEN_SAMPLER = "euler"
QWEN_SCHEDULER = "simple"
QWEN_MAX_REFS = 3
SERVER_WAIT_SECONDS = 120
GENERATION_WAIT_SECONDS = 7200
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tiff")


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


def _run(cmd, cwd=None):
    return subprocess.run(cmd, cwd=cwd).returncode


def ensure_comfyui(comfy_dir):
    if not os.path.exists(os.path.join(comfy_dir, "main.py")):
        print(f"Cloning ComfyUI into {comfy_dir}...")
        os.makedirs(os.path.dirname(comfy_dir), exist_ok=True)
        if _run(["git", "clone", "--depth", "1", COMFYUI_REPO, comfy_dir]) != 0:
            return False
    node_dir = os.path.join(comfy_dir, "custom_nodes", "ComfyUI-GGUF")
    if not os.path.exists(os.path.join(node_dir, "__init__.py")):
        print("Cloning ComfyUI-GGUF custom node (leejet fork, Qwen-Image 2.1 support)...")
        if _run(["git", "clone", "--depth", "1", GGUF_NODE_REPO, node_dir]) != 0:
            return False
    return True


def ensure_weights(comfy_dir):
    from huggingface_hub import hf_hub_download
    models_dir = os.path.join(comfy_dir, "models")
    targets = [
        (QWEN_GGUF_FILE, os.path.join(models_dir, "unet", QWEN_GGUF_NAME), os.path.join(models_dir, "unet"), "~4.6GB"),
        (QWEN_CLIP_FILE, os.path.join(models_dir, "text_encoders", QWEN_CLIP_NAME), os.path.join(models_dir), "~9.35GB"),
        (QWEN_VAE_FILE, os.path.join(models_dir, "vae", QWEN_VAE_NAME), os.path.join(models_dir), "~676MB"),
    ]
    for repo_file, target, local_dir, size in targets:
        if os.path.exists(target) and os.path.getsize(target) > 0:
            continue
        print(f"Downloading {repo_file} ({size}) from {QWEN_REPO_ID}...")
        os.makedirs(os.path.dirname(target), exist_ok=True)
        downloaded = hf_hub_download(repo_id=QWEN_REPO_ID, filename=repo_file, local_dir=local_dir)
        if os.path.abspath(downloaded) != os.path.abspath(target) and os.path.exists(downloaded):
            shutil.move(downloaded, target)
    return True


def free_port(start=8188):
    for port in range(start, start + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                try:
                    s.bind(("127.0.0.1", port))
                    return port
                except OSError:
                    continue
    raise RuntimeError("No free port found for ComfyUI server")


def wait_for_port(port, timeout):
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(1)
    return False


def launch_server(comfy_dir, port):
    import torch
    args = [sys.executable, os.path.join(comfy_dir, "main.py"),
            "--listen", "127.0.0.1", "--port", str(port), "--lowvram"]
    if not torch.cuda.is_available():
        args.append("--cpu")
        print("CUDA not available — ComfyUI will run on CPU (expect slow generation, 32GB RAM recommended)")
    log_file = open(os.path.join(comfy_dir, "voder_comfyui.log"), "w")
    proc = subprocess.Popen(args, cwd=comfy_dir, stdout=log_file, stderr=subprocess.STDOUT)
    return proc, log_file


def build_workflow(action, prompt, width, height, seed, steps, resolution_int, input_name=None, ref_names=None):
    workflow = {
        "10": {"inputs": {"unet_name": QWEN_GGUF_NAME}, "class_type": "UnetLoaderGGUF"},
        "11": {"inputs": {"clip_name": QWEN_CLIP_NAME, "type": "qwen_image"}, "class_type": "CLIPLoader"},
        "12": {"inputs": {"vae_name": QWEN_VAE_NAME}, "class_type": "VAELoader"},
        "15": {"inputs": {"model": ["10", 0], "device": "auto", "dtype": "default"}, "class_type": "QwenImage21Cache"},
        "3": {
            "inputs": {
                "seed": seed,
                "steps": steps,
                "cfg": QWEN_CFG,
                "sampler_name": QWEN_SAMPLER,
                "scheduler": QWEN_SCHEDULER,
                "denoise": 1.0,
                "model": ["15", 0],
                "positive": ["13", 0],
                "negative": ["13", 1],
                "latent_image": ["13", 2] if action == "edit" else ["14", 0],
            },
            "class_type": "KSampler",
        },
        "8": {"inputs": {"samples": ["3", 0], "vae": ["12", 0]}, "class_type": "VAEDecode"},
        "9": {"inputs": {"filename_prefix": "voder_eva_tti_overdose", "images": ["8", 0]}, "class_type": "SaveImage"},
        "13": {
            "inputs": {
                "prompt": prompt,
                "negative_prompt": "",
                "resolution": resolution_int,
                "clip": ["11", 0],
                "vae": ["12", 0],
            },
            "class_type": "TextEncodeQwenImage21",
        },
    }
    if action == "generate":
        workflow["14"] = {"inputs": {"width": width, "height": height, "batch_size": 1}, "class_type": "EmptyLatentImage"}
    if input_name:
        workflow["20"] = {"inputs": {"image": input_name}, "class_type": "LoadImage"}
        workflow["13"]["inputs"]["image_1"] = ["20", 0]
        ref_slot = 2
    else:
        ref_slot = 1
    for idx, ref_name in enumerate((ref_names or [])[:QWEN_MAX_REFS]):
        node_id = str(20 + idx + (1 if input_name else 0))
        workflow[node_id] = {"inputs": {"image": ref_name}, "class_type": "LoadImage"}
        workflow["13"]["inputs"][f"image_{ref_slot + idx}"] = [node_id, 0]
    return workflow


def stage_input(path, comfy_dir, tag):
    ext = os.path.splitext(path)[1].lower() or ".png"
    if ext not in IMAGE_EXTS:
        ext = ".png"
    stamp = time.strftime("%Y%m%d_%H%M%S")
    dest = os.path.join(comfy_dir, "input", f"voder_qwen_uc_{tag}_{stamp}{ext}")
    shutil.copyfile(path, dest)
    return os.path.basename(dest)


def submit_workflow(port, workflow):
    data = json.dumps({"prompt": workflow}).encode("utf-8")
    req = urllib.request.Request(f"http://127.0.0.1:{port}/prompt", data=data, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        details = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"ComfyUI rejected the workflow: {details}")


def wait_for_output(port, prompt_id, comfy_dir, timeout):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/history/{prompt_id}", timeout=30) as resp:
                history = json.loads(resp.read())
        except (urllib.error.URLError, socket.timeout):
            time.sleep(2)
            continue
        if prompt_id in history:
            entry = history[prompt_id]
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                raise RuntimeError("ComfyUI reported an execution error (check the log in the model folder)")
            outputs = entry.get("outputs", {})
            for node_output in outputs.values():
                for image_info in node_output.get("images", []):
                    if image_info.get("type") == "output":
                        subfolder = image_info.get("subfolder", "")
                        return os.path.join(comfy_dir, "output", subfolder, image_info["filename"])
            raise RuntimeError("ComfyUI finished but produced no output image")
        time.sleep(2)
    raise RuntimeError(f"Generation timed out after {timeout} seconds")


def _parse_target_size(resolution, image_path):
    width, height = None, None
    if resolution:
        try:
            parts = str(resolution).lower().split("x")
            width, height = int(parts[0]), int(parts[1])
        except Exception:
            width, height = None, None
            print(f"Warning: invalid resolution format '{resolution}', falling back to the input image size")
    if width is None and image_path:
        from PIL import Image
        with Image.open(image_path) as img:
            width, height = img.size
        if max(width, height) > 2752:
            scale = 2752 / max(width, height)
            width, height = int(width * scale), int(height * scale)
            print(f"Warning: input exceeds the 2752px max dimension, resized target to {width}x{height}")
    if width is None:
        width, height = 1024, 1024
    width = max(32, (width // 32) * 32)
    height = max(32, (height // 32) * 32)
    return width, height


def _resolve_int(width, height):
    return max(32, int(round(math.sqrt(width * height))))


def handle_generate(spec):
    from voders.DLCs.eva._paths import QWEN_IMAGE_UC_COMFYUI_DIR
    comfy_dir = QWEN_IMAGE_UC_COMFYUI_DIR
    if not ensure_comfyui(comfy_dir):
        write_result(False, error="Failed to clone ComfyUI or ComfyUI-GGUF (git must be available)")
        return 1
    ensure_weights(comfy_dir)
    prompt = spec["prompt"]
    output_path = spec["output_path"]
    seed = int(spec.get("seed", 0))
    steps = int(spec.get("num_inference_steps", QWEN_STEPS))
    width, height = _parse_target_size(spec.get("resolution"), None)
    references = spec.get("reference_paths") or []
    if len(references) > QWEN_MAX_REFS:
        print(f"Warning: Qwen-Image-2.1 UC supports max {QWEN_MAX_REFS} references, got {len(references)}. Using first {QWEN_MAX_REFS}.")
        references = references[:QWEN_MAX_REFS]
    ref_names = [stage_input(ref, comfy_dir, f"ref{i}") for i, ref in enumerate(references)]
    resolution_int = _resolve_int(width, height)
    workflow = build_workflow("generate", prompt, width, height, seed, steps, resolution_int, ref_names=ref_names)
    port = free_port()
    proc, log_file = launch_server(comfy_dir, port)
    try:
        if not wait_for_port(port, SERVER_WAIT_SECONDS):
            write_result(False, error=f"ComfyUI server failed to start within {SERVER_WAIT_SECONDS} seconds (log: {os.path.join(comfy_dir, 'voder_comfyui.log')})")
            return 1
        print(f"Generating image ({width}x{height}) with Qwen-Image-2.1 UC overdose...")
        response = submit_workflow(port, workflow)
        prompt_id = response["prompt_id"]
        produced = wait_for_output(port, prompt_id, comfy_dir, GENERATION_WAIT_SECONDS)
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        shutil.copyfile(produced, output_path)
        print(f"Image generated: {output_path}")
        write_result(True, output_path=output_path)
        return 0
    finally:
        proc.terminate()
        log_file.close()


def handle_edit(spec):
    from voders.DLCs.eva._paths import QWEN_IMAGE_UC_COMFYUI_DIR
    from PIL import Image
    comfy_dir = QWEN_IMAGE_UC_COMFYUI_DIR
    if not ensure_comfyui(comfy_dir):
        write_result(False, error="Failed to clone ComfyUI or ComfyUI-GGUF (git must be available)")
        return 1
    ensure_weights(comfy_dir)
    input_path = spec["input_path"]
    prompt = spec["prompt"]
    output_path = spec["output_path"]
    seed = int(spec.get("seed", 0))
    steps = int(spec.get("num_inference_steps", QWEN_STEPS))
    references = spec.get("reference_paths") or []
    if len(references) > QWEN_MAX_REFS:
        print(f"Warning: Qwen-Image-2.1 UC supports max {QWEN_MAX_REFS} references, got {len(references)}. Using first {QWEN_MAX_REFS}.")
        references = references[:QWEN_MAX_REFS]
    with Image.open(input_path) as img:
        input_size = img.size
    width, height = _parse_target_size(spec.get("resolution"), input_path)
    if spec.get("resolution") is None and max(input_size) != max(width, height):
        print(f"Warning: edit follows the input image aspect; sampling area set to {width}x{height}")
    input_name = stage_input(input_path, comfy_dir, "input")
    ref_names = [stage_input(ref, comfy_dir, f"ref{i}") for i, ref in enumerate(references)]
    resolution_int = _resolve_int(width, height)
    workflow = build_workflow("edit", prompt, width, height, seed, steps, resolution_int, input_name=input_name, ref_names=ref_names)
    port = free_port()
    proc, log_file = launch_server(comfy_dir, port)
    try:
        if not wait_for_port(port, SERVER_WAIT_SECONDS):
            write_result(False, error=f"ComfyUI server failed to start within {SERVER_WAIT_SECONDS} seconds (log: {os.path.join(comfy_dir, 'voder_comfyui.log')})")
            return 1
        print(f"Editing image ({width}x{height}) with Qwen-Image-2.1 UC overdose...")
        response = submit_workflow(port, workflow)
        prompt_id = response["prompt_id"]
        produced = wait_for_output(port, prompt_id, comfy_dir, GENERATION_WAIT_SECONDS)
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        shutil.copyfile(produced, output_path)
        print(f"Image edited: {output_path}")
        write_result(True, output_path=output_path)
        return 0
    finally:
        proc.terminate()
        log_file.close()


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


if __name__ == "__main__":
    sys.exit(main())
