import os
import sys
import json
import math
import shutil
import subprocess
import tempfile

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

_LYRA2_VENDORED_ROOT = os.path.join(_SRC_DIR, "lyra2")

SPEC_PATH = os.environ.get("EVA_SPEC_PATH")
RESULT_PATH = os.environ.get("EVA_RESULT_PATH")

LYRA2_REPO_ID = "nvidia/Lyra-2.0"
ZOOMGS_MODULE = "lyra_2._src.inference.lyra2_zoomgs_inference"
GS_RECON_MODULE = "lyra_2._src.inference.vipe_da3_gs_recon"
GS_RECON_CHUNKED_MODULE = "lyra_2._src.inference.vipe_da3_chunked_gs_recon"
LYRA2_FPS = 16
LYRA2_CHUNK_THRESHOLD = 500
LYRA2_RECON_CHUNK_SIZE = 128
LYRA2_RECON_CHUNK_OVERLAP = 10
LYRA2_OFFICIAL_FRAMES_ZOOM_IN = 81
LYRA2_OFFICIAL_FRAMES_ZOOM_OUT = 241


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


def ensure_weights(model_dir):
    from huggingface_hub import snapshot_download
    main_ckpt = os.path.join(model_dir, "checkpoints", "model", "model")
    done_marker = os.path.join(model_dir, "checkpoints", ".voder_download_complete")
    if os.path.isdir(main_ckpt) and os.listdir(main_ckpt):
        return True
    if os.path.exists(done_marker):
        os.remove(done_marker)
    print(f"Downloading Lyra 2.0 checkpoints (~70GB) from {LYRA2_REPO_ID} into {model_dir}...")
    snapshot_download(
        repo_id=LYRA2_REPO_ID,
        allow_patterns=["checkpoints/*"],
        local_dir=model_dir,
        token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
    )
    if not os.path.isdir(main_ckpt) or not os.listdir(main_ckpt):
        raise RuntimeError("Lyra 2.0 checkpoint download finished but the main model folder is missing or empty")
    with open(done_marker, "w") as f:
        f.write("ok")
    return True


def _resolve_frame_counts(duration):
    if duration is None or duration <= 0:
        return None, None, LYRA2_OFFICIAL_FRAMES_ZOOM_IN + LYRA2_OFFICIAL_FRAMES_ZOOM_OUT
    total = int(round(duration * LYRA2_FPS))
    k_in = max(1, int(round((total / 4.0 - 1) / 80.0)))
    n_in = 1 + 80 * k_in
    k_out = max(1, int(round((total - n_in - 1) / 80.0)))
    n_out = 1 + 80 * k_out
    if n_in + n_out != total:
        print(f"Note: frame counts snapped to the official 1+80k chunk rule (minimum 81 frames per direction) — zoom-in {n_in} + zoom-out {n_out} frames (~{(n_in + n_out) / LYRA2_FPS:.1f}s at {LYRA2_FPS} fps)")
    return n_in, n_out, n_in + n_out


def _env_for_subprocess():
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = _LYRA2_VENDORED_ROOT + (os.pathsep + existing if existing else "")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if os.path.exists(os.path.join(_SRC_DIR, "HF_TOKEN.txt")):
        with open(os.path.join(_SRC_DIR, "HF_TOKEN.txt"), "r") as f:
            content = f.read().strip()
            lines = [line.strip() for line in content.split("\n") if line.strip() and not line.strip().startswith("#")]
            if lines and not env.get("HF_TOKEN"):
                env["HF_TOKEN"] = lines[0]
                env["HUGGING_FACE_HUB_TOKEN"] = lines[0]
    return env


def _run_module(module, args, work_cwd):
    cmd = [sys.executable, "-m", module] + args
    print("\n" + "=" * 60)
    print("  Running: " + " ".join(cmd))
    print("=" * 60 + "\n", flush=True)
    return subprocess.run(cmd, cwd=work_cwd, env=_env_for_subprocess()).returncode


def _num_chunks_for(total_frames):
    if total_frames <= LYRA2_CHUNK_THRESHOLD:
        return 1
    stride = LYRA2_RECON_CHUNK_SIZE - LYRA2_RECON_CHUNK_OVERLAP
    return max(2, math.ceil((total_frames - LYRA2_RECON_CHUNK_OVERLAP) / stride))


def handle_generate(spec):
    from voders.DLCs.eva._paths import LYRA2_DIR
    input_path = os.path.abspath(spec["input_path"])
    output_dir = os.path.abspath(spec["output_path"])
    caption = spec["prompt"]
    seed = int(spec.get("seed", 1))
    fast = bool(spec.get("fast", False))
    trajectory = spec.get("trajectory")
    direction = spec.get("direction")
    strength = spec.get("strength")
    n_in, n_out, total_frames = _resolve_frame_counts(spec.get("duration"))

    ensure_weights(LYRA2_DIR)
    os.makedirs(output_dir, exist_ok=True)
    work_dir = tempfile.mkdtemp(prefix="voder_lyra2_")
    try:
        image_stem = os.path.splitext(os.path.basename(input_path))[0]
        videos_dir = os.path.join(work_dir, "videos")
        combined_video = os.path.join(videos_dir, f"{image_stem}.mp4")

        args = [
            "--input_image_path", input_path,
            "--prompt", caption,
            "--experiment", "lyra2",
            "--checkpoint_dir", "checkpoints/model",
            "--output_path", work_dir,
            "--seed", str(seed),
            "--offload",
            "--offload_when_prompt",
        ]
        if n_in is not None:
            args += ["--num_frames_zoom_in", str(n_in), "--num_frames_zoom_out", str(n_out)]
        if trajectory:
            args += ["--zoom_in_trajectory", trajectory, "--zoom_out_trajectory", trajectory]
        if direction:
            args += ["--zoom_in_direction", direction, "--zoom_out_direction", direction]
        if strength is not None:
            args += ["--zoom_in_strength", str(strength), "--zoom_out_strength", str(strength)]
        if fast:
            args += ["--use_dmd"]

        print(f"Step 1/2 — generating the exploration video (zoom-in + zoom-out camera trajectory{' , DMD fast mode' if fast else ''})...")
        if _run_module(ZOOMGS_MODULE, args, LYRA2_DIR) != 0:
            write_result(False, error="Lyra 2.0 video generation failed (check the step log above)")
            return 1
        if not os.path.isfile(combined_video):
            write_result(False, error=f"Lyra 2.0 finished but the combined video was not found at {combined_video}")
            return 1

        recon_dir = os.path.join(work_dir, "recon")
        recon_args = ["--input_video_path", combined_video, "--output_dir", recon_dir]
        module = GS_RECON_MODULE
        chunks = _num_chunks_for(total_frames)
        if chunks > 1:
            module = GS_RECON_CHUNKED_MODULE
            recon_args += ["--num_chunks", str(chunks), "--chunk_size", str(LYRA2_RECON_CHUNK_SIZE), "--chunk_overlap", str(LYRA2_RECON_CHUNK_OVERLAP)]
            print(f"Step 2/2 — reconstructing the 3D Gaussian Splatting scene ({total_frames} frames, {chunks} overlapping chunks)...")
        else:
            print("Step 2/2 — reconstructing the 3D Gaussian Splatting scene (VIPE poses + DA3 depth)...")
        if _run_module(module, recon_args, LYRA2_DIR) != 0:
            write_result(False, error="Lyra 2.0 Gaussian Splatting reconstruction failed (check the step log above)")
            return 1
        scene_ply = os.path.join(recon_dir, "reconstructed_scene.ply")
        flythrough = os.path.join(recon_dir, "gs_trajectory.mp4")
        if not os.path.isfile(scene_ply):
            write_result(False, error="Reconstruction finished but reconstructed_scene.ply was not produced")
            return 1

        shutil.copyfile(scene_ply, os.path.join(output_dir, "reconstructed_scene.ply"))
        if os.path.isfile(flythrough):
            shutil.copyfile(flythrough, os.path.join(output_dir, "camera_flythrough.mp4"))
        shutil.copyfile(combined_video, os.path.join(output_dir, "exploration_video.mp4"))
        print(f"Explorable scene saved: {output_dir}")
        write_result(True, output_path=output_dir, extra={"frames": total_frames})
        return 0
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def main():
    spec = load_spec()
    if spec is None:
        write_result(False, error="No spec provided")
        return 1
    action = spec.get("action")
    handlers = {"generate": handle_generate}
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
