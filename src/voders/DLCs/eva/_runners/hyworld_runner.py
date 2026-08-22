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


def _load_pipeline(model_dir):
    import torch
    from huggingface_hub import snapshot_download
    if not os.path.exists(os.path.join(model_dir, "hyworld2")):
        print(f"Downloading HY-World 2.0 to {model_dir} (large)...")
        snapshot_download(
            repo_id="Tencent-Hunyuan/HY-World-2.0",
            local_dir=model_dir,
            token=os.environ.get("HF_TOKEN"),
        )
    print("Loading HY-World 2.0 WorldStereo...")
    from hyworld2.worldgen.models.worldstereo_wrapper import WorldStereo
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    pipeline = WorldStereo.from_pretrained(model_dir, device=torch.device(device))
    print("HY-World 2.0 loaded.")
    return pipeline


def handle_generate(spec):
    import torch
    from voders.DLCs.eva._paths import HYWORLD_DIR
    pipeline = _load_pipeline(HYWORLD_DIR)
    prompt = spec["prompt"]
    output_path = spec["output_path"]
    seed = int(spec.get("seed", 0))
    references = spec.get("reference_paths") or []
    torch.manual_seed(seed)
    kwargs = {"prompt": prompt}
    if references:
        kwargs["reference_images"] = references
    print(f"Generating 3D world with HY-World 2.0...")
    result = pipeline(**kwargs)
    save_path = output_path.rsplit(".", 1)[0] + ".glb"
    output_saved = False
    for attr in ["mesh", "scene", "point_cloud", "gaussian", "meshes", "outputs", "video"]:
        if hasattr(result, attr):
            obj = getattr(result, attr)
            if obj is not None:
                if isinstance(obj, str) and os.path.exists(obj):
                    import shutil
                    shutil.move(obj, save_path)
                elif hasattr(obj, "export"):
                    obj.export(save_path)
                elif hasattr(obj, "save"):
                    obj.save(save_path)
                else:
                    import pickle
                    with open(save_path, "wb") as f:
                        pickle.dump(obj, f)
                print(f"World saved: {save_path}")
                output_saved = True
                break
    if not output_saved:
        write_result(False, error="HY-World produced no exportable mesh output — pipeline returned an unrecognized object type")
        return 1
    write_result(True, output_path=save_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
