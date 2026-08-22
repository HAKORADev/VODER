import os
import sys
import json
import tempfile

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
    handlers = {"animify": handle_animify}
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


def _prepare_src_root(spec):
    src_root = tempfile.mkdtemp(prefix="wan_animate_src_")
    pose_video = spec["pose_video"]
    ref_image = spec["reference_image"]
    src_pose = os.path.join(src_root, "src_pose.mp4")
    src_face = os.path.join(src_root, "src_face.mp4")
    src_ref = os.path.join(src_root, "src_ref.png")
    try:
        import shutil
        shutil.copy(pose_video, src_pose)
        shutil.copy(pose_video, src_face)
        shutil.copy(ref_image, src_ref)
    except Exception as e:
        write_result(False, error=f"Failed to stage source files: {e}")
        return None
    return src_root


def _load_pipeline(checkpoint_dir):
    import torch
    from huggingface_hub import snapshot_download
    if not os.path.exists(os.path.join(checkpoint_dir, "config.json")):
        print(f"Downloading Wan2.2-Animate-14B to {checkpoint_dir} (large)...")
        snapshot_download(
            repo_id="Wan-AI/Wan2.2-Animate-14B",
            local_dir=checkpoint_dir,
            token=os.environ.get("HF_TOKEN"),
        )
    print("Loading Wan2.2-Animate pipeline...")
    from wan22 import WanAnimate
    from wan22.configs.wan_animate_14B import animate_14B as cfg
    device_id = 0 if torch.cuda.is_available() else -1
    pipeline = WanAnimate(
        config=cfg,
        checkpoint_dir=checkpoint_dir,
        device_id=device_id,
        rank=0,
        t5_cpu=True,
        offload_model=True,
    )
    print("Wan2.2-Animate loaded.")
    return pipeline


def handle_animify(spec):
    import torch
    from voders.DLCs.eva._paths import ANIMATE_DIR
    pipeline = _load_pipeline(ANIMATE_DIR)
    src_root = _prepare_src_root(spec)
    if src_root is None:
        return 1
    output_path = spec["output_path"]
    prompt = spec.get("prompt", "")
    seed = int(spec.get("seed", -1))
    clip_len = int(spec.get("clip_len", 77))
    sampling_steps = int(spec.get("sampling_steps", 20))
    guide_scale = float(spec.get("guide_scale", 1.0))
    replace_flag = bool(spec.get("replace_flag", False))
    refert_num = int(spec.get("refert_num", 1))
    print(f"Generating animation with Wan2.2-Animate (clip_len={clip_len}, steps={sampling_steps})...")
    video = pipeline.generate(
        src_root_path=src_root,
        replace_flag=replace_flag,
        clip_len=clip_len,
        refert_num=refert_num,
        shift=5.0,
        sample_solver="dpm++",
        sampling_steps=sampling_steps,
        guide_scale=guide_scale,
        input_prompt=prompt,
        n_prompt="",
        seed=seed,
        offload_model=True,
    )
    if video is None:
        write_result(False, error="Wan2.2-Animate produced no output")
        return 1
    _save_video(video, output_path, fps=30)
    import shutil
    shutil.rmtree(src_root, ignore_errors=True)
    print(f"Animation saved: {output_path}")
    write_result(True, output_path=output_path)
    return 0


def _save_video(video, output_path, fps=30):
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


if __name__ == "__main__":
    sys.exit(main())
