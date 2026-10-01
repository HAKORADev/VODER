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
import uuid

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
QWEN_NBG_PROMPT_PREFIX = "This is an RGBA image with transparency."
QWEN_NBG_PROMPT_SUFFIX = "The image has alpha channel and the background is transparent."
NODE_LABELS = {
    "10": "Loading transformer",
    "11": "Loading text encoder",
    "12": "Loading VAE",
    "15": "Preparing model cache",
    "13": "Encoding prompt",
    "3": "Sampling",
    "8": "Decoding",
    "9": "Saving",
}


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


def submit_workflow(port, workflow, client_id):
    data = json.dumps({"prompt": workflow, "client_id": client_id}).encode("utf-8")
    req = urllib.request.Request(f"http://127.0.0.1:{port}/prompt", data=data, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        details = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"ComfyUI rejected the workflow: {details}")


def _fmt_secs(secs):
    secs = int(max(0, secs))
    return f"{secs // 60:02d}:{secs % 60:02d}"


def _connect_progress(port, client_id):
    try:
        from websocket import create_connection, WebSocketTimeoutException
    except ImportError:
        print("Warning: websocket-client is not installed in this env, live step progress is unavailable")
        return None, None
    try:
        ws = create_connection(f"ws://127.0.0.1:{port}/ws?clientId={client_id}", timeout=5)
        return ws, WebSocketTimeoutException
    except Exception as e:
        print(f"Warning: progress stream unavailable ({e}), live step progress is unavailable")
        return None, None


def _render_bar(label, value, total, started):
    elapsed = time.time() - started
    if value > 0:
        rate = elapsed / value
        eta = rate * (total - value)
        text = f"{label}: {value}/{total} [{_fmt_secs(elapsed)}<{_fmt_secs(eta)}, {rate:.2f}s/it]"
    else:
        text = f"{label}: 0/{total} [{_fmt_secs(elapsed)}<...]"
    print("\r" + text + " " * max(0, 24 - len(text)), end="", flush=True)


def _handle_progress_message(msg, prompt_id, state):
    if not isinstance(msg, dict):
        return state
    mtype = msg.get("type")
    data = msg.get("data") or {}
    if data.get("prompt_id") != prompt_id:
        return state
    if mtype == "execution_error":
        message = data.get("exception_message") or data.get("exception_type") or "unknown ComfyUI execution error"
        raise RuntimeError(f"ComfyUI execution error: {message}")
    if mtype == "executing":
        label = NODE_LABELS.get(str(data.get("node")))
        if label and label != state.get("bar") and label != state.get("stage"):
            if state.get("bar") is not None:
                print()
                state["bar"] = None
            print(f"{label}...")
            state["stage"] = label
        return state
    if mtype != "progress":
        return state
    node = str(data.get("node"))
    value = int(data.get("value") or 0)
    total = int(data.get("max") or 0)
    if total <= 0:
        return state
    label = NODE_LABELS.get(node, f"Node {node}")
    if state.get("bar") != label:
        if state.get("bar") is not None:
            print()
        if state.get("stage") != label:
            print(f"{label}...")
        state["bar"] = label
        state["stage"] = label
        state["started"] = time.time()
    _render_bar(label, value, total, state["started"])
    return state


def wait_for_output(port, prompt_id, comfy_dir, timeout, ws=None, ws_timeout_exc=None):
    deadline = time.time() + timeout
    next_poll = 0.0
    state = {"bar": None, "stage": None, "started": 0.0}
    while time.time() < deadline:
        if ws is not None:
            try:
                frame = ws.recv()
                if isinstance(frame, str) and frame:
                    try:
                        _handle_progress_message(json.loads(frame), prompt_id, state)
                    except json.JSONDecodeError:
                        pass
            except ws_timeout_exc:
                pass
            except RuntimeError:
                raise
            except Exception:
                if state.get("bar") is not None:
                    print()
                    state["bar"] = None
                try:
                    ws.close()
                except Exception:
                    pass
                ws = None
                print("Warning: progress stream lost, continuing on history polling")
        now = time.time()
        if now >= next_poll:
            next_poll = now + 2
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/history/{prompt_id}", timeout=30) as resp:
                    history = json.loads(resp.read())
            except (urllib.error.URLError, socket.timeout):
                continue
            if prompt_id in history:
                entry = history[prompt_id]
                status = entry.get("status", {})
                if status.get("status_str") == "error":
                    raise RuntimeError("ComfyUI reported an execution error (check the log in the model folder)")
                produced = None
                for node_output in entry.get("outputs", {}).values():
                    for image_info in node_output.get("images", []):
                        if image_info.get("type") == "output":
                            subfolder = image_info.get("subfolder", "")
                            produced = os.path.join(comfy_dir, "output", subfolder, image_info["filename"])
                if produced:
                    if state.get("bar") is not None:
                        print()
                    return produced
                raise RuntimeError("ComfyUI finished but produced no output image")
        time.sleep(0.2)
    if state.get("bar") is not None:
        print()
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


def _execute_workflow(comfy_dir, workflow, stage_msg):
    client_id = str(uuid.uuid4())
    port = free_port()
    proc, log_file = launch_server(comfy_dir, port)
    try:
        if not wait_for_port(port, SERVER_WAIT_SECONDS):
            write_result(False, error=f"ComfyUI server failed to start within {SERVER_WAIT_SECONDS} seconds (log: {os.path.join(comfy_dir, 'voder_comfyui.log')})")
            return None
        print(stage_msg)
        response = submit_workflow(port, workflow, client_id)
        prompt_id = response["prompt_id"]
        ws, ws_timeout_exc = _connect_progress(port, client_id)
        try:
            return wait_for_output(port, prompt_id, comfy_dir, GENERATION_WAIT_SECONDS, ws, ws_timeout_exc)
        except RuntimeError as e:
            write_result(False, error=str(e))
            return None
        finally:
            if ws is not None:
                try:
                    ws.close()
                except Exception:
                    pass
    finally:
        proc.terminate()
        log_file.close()


def _warn_if_opaque(path):
    try:
        from PIL import Image
        with Image.open(path) as img:
            if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
                alpha = img.convert("RGBA").getchannel("A")
                if alpha.getextrema()[0] >= 250:
                    print("Warning: the output came back fully opaque — the model did not apply transparency for this prompt")
    except Exception:
        pass


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
    produced = _execute_workflow(comfy_dir, workflow, f"Generating image ({width}x{height}) with Qwen-Image-2.1 UC overdose...")
    if produced is None:
        return 1
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    shutil.copyfile(produced, output_path)
    print(f"Image generated: {output_path}")
    write_result(True, output_path=output_path)
    return 0


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
    produced = _execute_workflow(comfy_dir, workflow, f"Editing image ({width}x{height}) with Qwen-Image-2.1 UC overdose...")
    if produced is None:
        return 1
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    shutil.copyfile(produced, output_path)
    print(f"Image edited: {output_path}")
    write_result(True, output_path=output_path)
    return 0


def handle_generate_nbg(spec):
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
    nbg_prompt = f"{QWEN_NBG_PROMPT_PREFIX} {prompt}. {QWEN_NBG_PROMPT_SUFFIX}"
    resolution_int = _resolve_int(width, height)
    workflow = build_workflow("generate", nbg_prompt, width, height, seed, steps, resolution_int)
    produced = _execute_workflow(comfy_dir, workflow, f"Generating transparent image ({width}x{height}) with Qwen-Image-2.1 UC overdose...")
    if produced is None:
        return 1
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    shutil.copyfile(produced, output_path)
    _warn_if_opaque(output_path)
    print(f"Transparent image generated: {output_path}")
    write_result(True, output_path=output_path)
    return 0


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
