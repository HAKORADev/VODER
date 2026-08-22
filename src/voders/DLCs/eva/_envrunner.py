import os
import sys
import json
import subprocess
import tempfile

_EVA_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.dirname(_EVA_DIR)))
ENVS_DIR = os.path.join(_SRC_DIR, "envs")

if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

EVA_RUNNERS = {
    "flux2": "flux2",
    "h3": "h3",
    "vace": "vace",
    "animate": "animate",
    "s2v": "animate",
    "hyworld": "hyworld",
    "trellis": "trellis",
    "sam3": "sam3",
    "siglip2": "siglip2",
}


def _venv_name(model_name):
    return EVA_RUNNERS.get(model_name, model_name)


def venv_python(model_name):
    env_dir = os.path.join(ENVS_DIR, _venv_name(model_name))
    if sys.platform == "win32":
        return os.path.join(env_dir, "Scripts", "python.exe")
    return os.path.join(env_dir, "bin", "python")


def venv_exists(model_name):
    return os.path.exists(venv_python(model_name))


def runner_script(model_name):
    script = os.path.join(_EVA_DIR, "_runners", f"{model_name}_runner.py")
    if not os.path.exists(script):
        raise FileNotFoundError(f"Runner script not found for model '{model_name}': {script}")
    return script


def run_in_venv(model_name, spec, timeout=None):
    if model_name not in EVA_RUNNERS:
        raise ValueError(f"Unknown Eva model: {model_name}. Known: {list(EVA_RUNNERS.keys())}")
    if not venv_exists(model_name):
        return {
            "success": False,
            "error": (
                f"Python environment for '{model_name}' is not set up. "
                f"Run: python setup.py --envs {_venv_name(model_name)}  "
                f"(or: python setup.py --envs all)"
            ),
        }
    py = venv_python(model_name)
    script = runner_script(model_name)
    spec_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", prefix=f"eva_{model_name}_spec_", delete=False
    )
    json.dump(spec, spec_file)
    spec_file.close()
    spec_path = spec_file.name
    result_path = spec_path.replace("_spec_", "_result_")
    env = os.environ.copy()
    env["PYTHONPATH"] = _SRC_DIR + os.pathsep + env.get("PYTHONPATH", "")
    env["EVA_SPEC_PATH"] = spec_path
    env["EVA_RESULT_PATH"] = result_path
    if os.path.exists(os.path.join(_SRC_DIR, "HF_TOKEN.txt")):
        with open(os.path.join(_SRC_DIR, "HF_TOKEN.txt"), "r") as f:
            content = f.read().strip()
            lines = [line.strip() for line in content.split("\n") if line.strip() and not line.strip().startswith("#")]
            if lines:
                env["HF_TOKEN"] = lines[0]
                env["HUGGING_FACE_HUB_TOKEN"] = lines[0]
    try:
        proc = subprocess.run(
            [py, script],
            env=env,
            timeout=timeout,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if proc.stdout:
            print(proc.stdout, end="")
        if os.path.exists(result_path):
            with open(result_path, "r") as f:
                result = json.load(f)
            return result
        return {
            "success": False,
            "error": f"Runner did not produce a result file (exit code {proc.returncode}).",
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "error": f"Runner timed out after {timeout}s."}
    except Exception as e:
        return {"success": False, "error": f"Runner invocation failed: {e}"}
    finally:
        for p in (spec_path, result_path):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass
