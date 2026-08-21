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
    handlers = {"generate": handle_generate, "edit": handle_edit}
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
    if not os.path.exists(os.path.join(model_dir, "modular_model_index.json")):
        print(f"Downloading MiniMax H3 to {model_dir} (large)...")
        snapshot_download(
            repo_id="MiniMaxAI/MiniMax-H3",
            local_dir=model_dir,
            token=os.environ.get("HF_TOKEN"),
        )
    print("Loading MiniMax H3 pipeline...")
    from h3 import (
        MiniMaxH3ModularPipeline,
        AutoencoderKLMiniMaxH3,
        AutoencoderKLMiniMaxH3Audio,
        MiniMaxH3Transformer3DModel,
        MiniMaxH3Scheduler,
    )
    from h3.modular_pipelines.minimax_h3.modular_blocks_minimax_h3 import MiniMaxH3Blocks
    from diffusers.guiders import ClassifierFreeGuidance
    from transformers import Qwen3VLForConditionalGeneration, Qwen2TokenizerFast, Qwen3VLProcessor
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    pipe = MiniMaxH3ModularPipeline()
    pipe.tokenizer = Qwen2TokenizerFast.from_pretrained(os.path.join(model_dir, "tokenizer"))
    pipe.text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
        os.path.join(model_dir, "text_encoder"),
        torch_dtype=dtype,
    ).to(device)
    pipe.processor = Qwen3VLProcessor.from_pretrained(os.path.join(model_dir, "processor"))
    pipe.vae = AutoencoderKLMiniMaxH3.from_pretrained(
        os.path.join(model_dir, "vae"),
        torch_dtype=dtype,
    ).to(device)
    pipe.audio_vae = AutoencoderKLMiniMaxH3Audio.from_pretrained(
        os.path.join(model_dir, "audio_vae"),
        torch_dtype=dtype,
    ).to(device)
    pipe.transformer = MiniMaxH3Transformer3DModel.from_pretrained(
        os.path.join(model_dir, "transformer"),
        torch_dtype=dtype,
    ).to(device)
    if os.path.exists(os.path.join(model_dir, "transformer_ref")):
        pipe.transformer_ref = MiniMaxH3Transformer3DModel.from_pretrained(
            os.path.join(model_dir, "transformer_ref"),
            torch_dtype=dtype,
        ).to(device)
    pipe.scheduler = MiniMaxH3Scheduler.from_pretrained(os.path.join(model_dir, "scheduler"))
    pipe.audio_scheduler = MiniMaxH3Scheduler.from_pretrained(os.path.join(model_dir, "audio_scheduler"))
    pipe.guider = ClassifierFreeGuidance(2.5)
    pipe._blocks = MiniMaxH3Blocks()
    print("MiniMax H3 loaded.")
    return pipe, device


def handle_generate(spec):
    import torch
    import numpy as np
    from voders.DLCs.eva._paths import H3_DIR
    pipe, device = _load_pipeline(H3_DIR)
    prompt = spec["prompt"]
    output_path = spec["output_path"]
    duration = int(spec.get("duration", 10))
    seed = int(spec.get("seed", 0))
    image_refs = spec.get("image_refs") or []
    video_refs = spec.get("video_refs") or []
    audio_refs = spec.get("audio_refs") or []
    print(f"Generating video with MiniMax H3 (up to {duration}s)...")
    generator = torch.Generator(device=device).manual_seed(seed)
    kwargs = dict(
        prompt=prompt,
        generator=generator,
        output="videos",
    )
    if image_refs:
        kwargs["images"] = image_refs
    if video_refs:
        kwargs["videos"] = video_refs
    if audio_refs:
        kwargs["audios"] = audio_refs
    output = pipe(**kwargs)
    video = output
    if isinstance(video, torch.Tensor):
        video = video.cpu().numpy()
    if isinstance(video, list) and len(video) > 0:
        video = video[0]
    if isinstance(video, dict):
        video = video.get("video", video.get("videos", None))
    if video is None:
        write_result(False, error="H3 produced no output")
        return 1
    if isinstance(video, bytes):
        with open(output_path, "wb") as f:
            f.write(video)
    elif isinstance(video, np.ndarray):
        import imageio
        if video.ndim == 4:
            video = video[0]
        if video.dtype != np.uint8:
            video = ((video + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
        if video.shape[-1] in (3, 4):
            video = video.transpose(0, 2, 3, 1)
        elif video.shape[0] in (3, 4):
            video = video.transpose(1, 2, 0)
        imageio.mimsave(output_path, video, fps=24)
    else:
        if hasattr(video, "save"):
            video.save(output_path)
        else:
            print(f"Warning: unknown output type {type(video)}, trying pickle...")
            with open(output_path, "wb") as f:
                import pickle
                pickle.dump(video, f)
    print(f"Video generated: {output_path}")
    write_result(True, output_path=output_path)
    return 0


def handle_edit(spec):
    write_result(False, error="H3 does not support edit; use VACE for video editing.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
