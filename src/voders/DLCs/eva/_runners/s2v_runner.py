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
    handlers = {"lipsync": handle_lipsync}
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


def _load_pipeline(checkpoint_dir):
    import torch
    from huggingface_hub import snapshot_download
    if not os.path.exists(os.path.join(checkpoint_dir, "config.json")):
        print(f"Downloading Wan2.2-S2V-14B to {checkpoint_dir} (large)...")
        snapshot_download(
            repo_id="Wan-AI/Wan2.2-S2V-14B",
            local_dir=checkpoint_dir,
            token=os.environ.get("HF_TOKEN"),
        )
    print("Loading Wan2.2-S2V pipeline...")
    from wan22 import WanS2V
    from wan22.configs.wan_s2v_14B import s2v_14B as cfg
    device_id = 0 if torch.cuda.is_available() else -1
    pipeline = WanS2V(
        config=cfg,
        checkpoint_dir=checkpoint_dir,
        device_id=device_id,
        rank=0,
        t5_cpu=True,
        convert_model_dtype=False,
    )
    print("Wan2.2-S2V loaded.")
    return pipeline


def handle_lipsync(spec):
    import torch
    from voders.DLCs.eva._paths import S2V_DIR
    pipeline = _load_pipeline(S2V_DIR)
    ref_image_path = spec["reference_image"]
    audio_path = spec["audio_path"]
    output_path = spec["output_path"]
    prompt = spec.get("prompt", "")
    seed = int(spec.get("seed", -1))
    infer_frames = int(spec.get("infer_frames", 80))
    sampling_steps = int(spec.get("sampling_steps", 40))
    guide_scale = float(spec.get("guide_scale", 4.5))
    init_first_frame = bool(spec.get("init_first_frame", True))
    enable_tts = bool(spec.get("enable_tts", False))
    print(f"Generating lip-sync video with Wan2.2-S2V (frames={infer_frames}, steps={sampling_steps})...")
    video = pipeline.generate(
        input_prompt=prompt,
        ref_image_path=ref_image_path,
        audio_path=audio_path,
        enable_tts=enable_tts,
        tts_prompt_audio=None,
        tts_prompt_text=None,
        tts_text=None,
        num_repeat=1,
        pose_video=None,
        max_area=720 * 1280,
        infer_frames=infer_frames,
        shift=3.0,
        sample_solver="unipc",
        sampling_steps=sampling_steps,
        guide_scale=guide_scale,
        n_prompt="",
        seed=seed,
        offload_model=True,
        init_first_frame=init_first_frame,
    )
    if video is None:
        write_result(False, error="Wan2.2-S2V produced no output")
        return 1
    _save_video(video, output_path, fps=24)
    try:
        _merge_audio(output_path, audio_path)
    except Exception as e:
        print(f"Warning: audio merge failed: {e}")
    print(f"Lip-sync video saved: {output_path}")
    write_result(True, output_path=output_path)
    return 0


def _save_video(video, output_path, fps=24):
    import numpy as np
    import imageio
    import torch
    if isinstance(video, torch.Tensor):
        video = video.cpu().numpy()
    if isinstance(video, list):
        video = video[0] if video else None
    if video is None:
        raise ValueError("Empty video tensor")
    if video.ndim == 4:
        video = video[0] if video.shape[0] == 3 else video
    if video.ndim == 3:
        video = video
    if video.dtype != np.uint8:
        video = ((video + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
    if video.shape[0] == 3:
        video = video.transpose(1, 2, 0)
    imageio.mimsave(output_path, video, fps=fps)


def _merge_audio(video_path, audio_path):
    import subprocess
    import tempfile
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-i", audio_path,
         "-c:v", "copy", "-c:a", "aac", "-shortest", tmp],
        capture_output=True, check=True
    )
    import shutil
    shutil.move(tmp, video_path)


if __name__ == "__main__":
    sys.exit(main())
