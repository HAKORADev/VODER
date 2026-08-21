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
    handlers = {"objectify": handle_objectify, "edit": handle_edit}
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


def _load_image_to_3d_pipeline(model_dir):
    import torch
    from huggingface_hub import snapshot_download
    if not os.path.exists(os.path.join(model_dir, "pipeline.json")):
        print(f"Downloading TRELLIS.2 to {model_dir} (~14GB)...")
        snapshot_download(
            repo_id="microsoft/TRELLIS.2-4B",
            local_dir=model_dir,
            token=os.environ.get("HF_TOKEN"),
        )
    print("Loading TRELLIS.2 image-to-3D pipeline...")
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(model_dir)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    pipeline.to(device)
    print("TRELLIS.2 image-to-3D loaded.")
    return pipeline


def _load_texturing_pipeline(model_dir):
    import torch
    from huggingface_hub import snapshot_download
    if not os.path.exists(os.path.join(model_dir, "texturing_pipeline.json")):
        print(f"Downloading TRELLIS.2 texturing config to {model_dir}...")
        snapshot_download(
            repo_id="microsoft/TRELLIS.2-4B",
            local_dir=model_dir,
            token=os.environ.get("HF_TOKEN"),
            allow_patterns=["texturing_pipeline.json", "shape_slat_encoder/*", "tex_slat_flow_model_512/*", "tex_slat_flow_model_1024/*", "tex_slat_decoder/*"],
        )
    print("Loading TRELLIS.2 texturing pipeline...")
    from trellis2.pipelines import Trellis2TexturingPipeline
    pipeline = Trellis2TexturingPipeline.from_pretrained(model_dir, config_file="texturing_pipeline.json")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    pipeline.to(device)
    print("TRELLIS.2 texturing loaded.")
    return pipeline


def handle_objectify(spec):
    import torch
    from PIL import Image
    from voders.DLCs.eva._paths import TRELLIS_DIR
    pipeline = _load_image_to_3d_pipeline(TRELLIS_DIR)
    input_path = spec["input_path"]
    output_path = spec["output_path"]
    seed = int(spec.get("seed", 0))
    print(f"Converting image to 3D object with TRELLIS.2...")
    image = Image.open(input_path).convert("RGB")
    torch.manual_seed(seed)
    outputs = pipeline.run(image=image, num_samples=1, seed=seed)
    if not isinstance(outputs, list):
        outputs = [outputs]
    mesh = outputs[0]
    save_path = output_path.rsplit(".", 1)[0] + ".glb"
    _export_glb(pipeline, mesh, save_path, resolution=1024)
    print(f"3D object saved: {save_path}")
    write_result(True, output_path=save_path)
    return 0


def handle_edit(spec):
    import torch
    from PIL import Image
    import trimesh
    from voders.DLCs.eva._paths import TRELLIS_DIR
    pipeline = _load_texturing_pipeline(TRELLIS_DIR)
    input_path = spec["input_path"]
    reference_image = spec["reference_image"]
    output_path = spec["output_path"]
    seed = int(spec.get("seed", 0))
    print(f"Retexturing 3D object with TRELLIS.2...")
    mesh = trimesh.load(input_path, force='mesh')
    image = Image.open(reference_image).convert("RGB")
    torch.manual_seed(seed)
    out_mesh = pipeline.run(mesh=mesh, image=image, seed=seed, texture_size=2048)
    save_path = output_path.rsplit(".", 1)[0] + ".glb"
    out_mesh.export(save_path)
    print(f"Retextured object saved: {save_path}")
    write_result(True, output_path=save_path)
    return 0


def _export_glb(pipeline, mesh, save_path, resolution=1024, decimation_target=1000000, texture_size=1024):
    import o_voxel
    vertices = mesh.vertices
    faces = mesh.faces
    if hasattr(vertices, "cpu"):
        vertices = vertices.cpu()
    if hasattr(faces, "cpu"):
        faces = faces.cpu()
    attrs = mesh.attrs
    if hasattr(attrs, "cpu"):
        attrs = attrs.cpu()
    coords = mesh.coords
    if hasattr(coords, "cpu"):
        coords = coords.cpu()
    glb = o_voxel.postprocess.to_glb(
        vertices=vertices,
        faces=faces,
        attr_volume=attrs,
        coords=coords,
        attr_layout=pipeline.pbr_attr_layout,
        grid_size=resolution,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=decimation_target,
        texture_size=texture_size,
        remesh=True,
        remesh_band=1,
        remesh_project=0,
        use_tqdm=True,
    )
    glb.export(save_path)


if __name__ == "__main__":
    sys.exit(main())
