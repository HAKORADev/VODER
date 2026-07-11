#!/usr/bin/env python3

import subprocess
import sys
import os

REQUIREMENTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements.txt")


def run(cmd, env=None, check=True):
    print(f"\n{'='*60}")
    print(f"  Running: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd, env=env)
    if check and result.returncode != 0:
        print(f"\n  ERROR: Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    return result.returncode


def has_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        pass
    try:
        result = subprocess.run([sys.executable, "-c", "import torch; print(torch.cuda.is_available())"],
                                capture_output=True, text=True)
        return result.stdout.strip() == "True"
    except Exception:
        return False


def is_llama_cpp_installed_with_cuda():
    try:
        import llama_cpp
        import glob
        spec = llama_cpp.__spec__
        if spec and spec.origin:
            pkg_dir = os.path.dirname(spec.origin)
            so_files = glob.glob(os.path.join(pkg_dir, "lib", "*.so")) + \
                       glob.glob(os.path.join(pkg_dir, "lib", "*.dylib")) + \
                       glob.glob(os.path.join(pkg_dir, "lib", "*.dll"))
            for so in so_files:
                try:
                    with open(so, "rb") as f:
                        content = f.read()
                        if b"cuda" in content.lower() or b"ggml_cuda" in content.lower():
                            return True
                except Exception:
                    pass
    except Exception:
        pass
    return False


def get_cuda_version():
    try:
        import torch
        if torch.cuda.is_available():
            version = torch.version.cuda
            if version:
                major, minor = version.split(".")[:2]
                return f"cu{major}{minor}"
    except Exception:
        pass
    try:
        import subprocess as _sp
        result = _sp.run(["nvcc", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            for line in result.stdout.split("\n"):
                if "release" in line.lower():
                    parts = line.split("release")[1].split(",")
                    if parts:
                        ver = parts[0].strip().split(".")
                        if len(ver) >= 2:
                            return f"cu{ver[0]}{ver[1]}"
    except Exception:
        pass
    return "cu124"


def install_llama_cpp_with_cuda():
    cuda_ver = get_cuda_version()
    print(f"\n  Installing llama-cpp-python with CUDA support ({cuda_ver})...")

    print("  Step 1: Uninstalling existing llama-cpp-python...")
    run([sys.executable, "-m", "pip", "uninstall", "llama-cpp-python", "-y"],
        check=False)

    import urllib.request
    import platform as _plat

    wheel_base = f"https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.33-{cuda_ver}"
    if _plat.system() == "Windows":
        wheel_url = f"{wheel_base}/llama_cpp_python-0.3.33-py3-none-win_amd64.whl"
    else:
        wheel_url = f"{wheel_base}/llama_cpp_python-0.3.33-py3-none-manylinux_2_35_x86_64.whl"

    print(f"  Step 2: Downloading CUDA wheel directly: {wheel_url}")
    wheel_path = "/tmp/llama_cpp_python_cuda.whl"
    try:
        urllib.request.urlretrieve(wheel_url, wheel_path)
        print(f"  Downloaded {os.path.getsize(wheel_path) / 1024 / 1024:.1f} MB")
        ret = run([sys.executable, "-m", "pip", "install", wheel_path, "--force-reinstall"],
                  check=False)
    except Exception as e:
        print(f"  Direct download failed: {e}")
        print(f"  Trying pip install with --find-links...")
        wheel_index = f"https://abetlen.github.io/llama-cpp-python/whl/{cuda_ver}/llama-cpp-python/"
        ret = run([sys.executable, "-m", "pip", "install", "llama-cpp-python",
                   "--no-cache-dir", "--find-links", wheel_index, "--no-index"],
                  check=False)
        if ret != 0:
            print(f"  --find-links also failed. Trying source build...")
            env = {**os.environ, "CMAKE_ARGS": "-DGGML_CUDA=on", "FORCE_CMAKE": "1"}
            ret = run([sys.executable, "-m", "pip", "install", "llama-cpp-python",
                       "--upgrade", "--force-reinstall", "--no-cache-dir"],
                      env=env, check=False)

    try:
        import llama_cpp
        import glob
        spec = llama_cpp.__spec__
        if spec and spec.origin:
            pkg_dir = os.path.dirname(spec.origin)
            so_files = glob.glob(os.path.join(pkg_dir, "lib", "*.so")) + \
                       glob.glob(os.path.join(pkg_dir, "lib", "*.dylib")) + \
                       glob.glob(os.path.join(pkg_dir, "lib", "*.dll"))
            cuda_found = False
            for so in so_files:
                try:
                    with open(so, "rb") as f:
                        content = f.read()
                        if b"cuda" in content.lower() or b"ggml_cuda" in content.lower():
                            cuda_found = True
                            break
                except Exception:
                    pass
            if cuda_found:
                print("  CUDA support VERIFIED in installed library.")
            else:
                print("  WARNING: CUDA support NOT found in installed library!")
                print("  Trying source build as last resort...")
                env = {**os.environ, "CMAKE_ARGS": "-DGGML_CUDA=on", "FORCE_CMAKE": "1"}
                run([sys.executable, "-m", "pip", "install", "llama-cpp-python",
                     "--upgrade", "--force-reinstall", "--no-cache-dir"],
                    env=env, check=False)
    except Exception:
        pass

    return ret


def install_llama_cpp_cpu():
    print("\n  Installing llama-cpp-python (CPU-only)...")
    return run([sys.executable, "-m", "pip", "install", "llama-cpp-python", "--upgrade", "--force-reinstall", "--no-cache-dir"],
               check=False)


def main():
    print("""
============================================================
  VODER Setup
============================================================
""")

    print("[1/4] Installing requirements.txt...")
    run([sys.executable, "-m", "pip", "install", "-r", REQUIREMENTS_FILE])

    print("\n[2/4] Installing protobuf 5.29.6 (fix for descript-audiotools)...")
    run([sys.executable, "-m", "pip", "install", "--upgrade", "protobuf==5.29.6"])

    cuda_available = has_cuda()
    print(f"\n[3/4] CUDA detection: {'CUDA GPU DETECTED' if cuda_available else 'No CUDA GPU (CPU mode)'}")

    already_installed_with_cuda = is_llama_cpp_installed_with_cuda()
    if cuda_available and not already_installed_with_cuda:
        print("  CUDA GPU detected but llama-cpp-python lacks CUDA support.")
        ret = install_llama_cpp_with_cuda()
        if ret != 0:
            print("  CUDA build failed. Falling back to CPU-only llama-cpp-python...")
            install_llama_cpp_cpu()
        else:
            print("  llama-cpp-python installed with CUDA support successfully.")
    elif not cuda_available:
        print("  No CUDA GPU detected. Installing CPU-only llama-cpp-python...")
        install_llama_cpp_cpu()
    else:
        print("  llama-cpp-python already has CUDA support. Skipping.")

    print("\n[4/4] Verifying installation...")
    try:
        import llama_cpp
        print(f"  llama-cpp-python: {llama_cpp.__version__ if hasattr(llama_cpp, '__version__') else 'installed'}")
    except ImportError:
        print("  WARNING: llama-cpp-python not found. VADAR Lite will not work.")

    try:
        import torch
        if torch.cuda.is_available():
            print(f"  PyTorch CUDA: {torch.cuda.get_device_name(0)}")
        else:
            print("  PyTorch CUDA: not available (CPU mode)")
    except ImportError:
        print("  PyTorch: not found")

    print("""
============================================================
  VODER Setup Complete!
============================================================

  Quick start:
    python voder.py vadar "hello there"
    python voder.py cli

  For VADAR heavy (overdose, multimodal):
    python voder.py overdose vadar "hello there"

  See docs/Guide.md for the full guide.
============================================================
""")


if __name__ == "__main__":
    main()
