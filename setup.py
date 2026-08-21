#!/usr/bin/env python3

import argparse
import os
import platform
import shutil
import subprocess
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
REQUIREMENTS_FILE = os.path.join(REPO_ROOT, "requirements.txt")
SRC_DIR = os.path.join(REPO_ROOT, "src")
ENVS_DIR = os.path.join(SRC_DIR, "envs")

EVA_ENVS = {
    "flux2":   "Flux 2 Dev / Klein 9B (image gen / edit / nbg / mini)",
    "h3":      "MiniMax H3 (video gen)",
    "vace":    "Wan 2.1 VACE 14B (video edit)",
    "animate": "Wan 2.2 Animate 14B + S2V 14B (video animify + lipsync)",
    "hyworld": "Tencent HY-World 2.0 (world gen / edit)",
    "trellis": "Microsoft TRELLIS.2 (image to 3D)",
    "sam3":    "Meta SAM 3.1 (segmentation)",
    "siglip2": "SigLIP 2 giant (vision encoder)",
}


def run(cmd, env=None, check=True, shell=False, capture=False):
    print(f"\n{'='*60}")
    print(f"  Running: {cmd if isinstance(cmd, str) else ' '.join(str(c) for c in cmd)}")
    print(f"{'='*60}\n")
    if capture:
        result = subprocess.run(cmd, env=env, shell=shell, capture_output=True, text=True)
    else:
        result = subprocess.run(cmd, env=env, shell=shell)
    if check and result.returncode != 0:
        print(f"\n  WARNING: Command returned exit code {result.returncode}")
    return result


def is_linux():
    return platform.system() == "Linux"


def is_windows():
    return platform.system() == "Windows"


def is_macos():
    return platform.system() == "Darwin"


def command_exists(cmd):
    return shutil.which(cmd) is not None


def have_sudo():
    if not is_linux():
        return False
    if os.geteuid() == 0:
        return False
    if not shutil.which("sudo"):
        return False
    try:
        r = subprocess.run(["sudo", "-n", "true"], capture_output=True, timeout=3)
        return r.returncode == 0
    except Exception:
        return False


def package_manager():
    if command_exists("apt-get") or command_exists("apt"):
        return "apt"
    if command_exists("pacman"):
        return "pacman"
    if command_exists("dnf"):
        return "dnf"
    if command_exists("yum"):
        return "yum"
    if command_exists("zypper"):
        return "zypper"
    if command_exists("apk"):
        return "apk"
    return None


def try_install_system_packages():
    print("\n[SYSTEM] Checking system packages...")

    packages_needed = []
    if not command_exists("ffmpeg"):
        packages_needed.append("ffmpeg")
    if not command_exists("sox"):
        packages_needed.append("sox")
    if not command_exists("zstd"):
        packages_needed.append("zstd")
    if is_linux() and not command_exists("lspci") and not command_exists("lshw"):
        packages_needed.append("lshw")
    if not command_exists("git"):
        packages_needed.append("git")
    if not command_exists("curl"):
        packages_needed.append("curl")

    packages_needed = list(dict.fromkeys(packages_needed))
    if not packages_needed:
        print("  All system packages already installed.")
        return

    print(f"  Missing: {', '.join(packages_needed)}")

    if is_linux():
        pm = package_manager()
        if pm is None:
            print("  WARNING: Unknown package manager. Please install manually: " + ", ".join(packages_needed))
            return
        can_sudo = have_sudo()
        if can_sudo:
            print(f"  sudo: available (will use sudo for {pm})")
        elif os.geteuid() == 0:
            print(f"  sudo: not needed (running as root, using {pm} directly)")
        elif not shutil.which("sudo"):
            print(f"  sudo: not installed (trying {pm} without sudo — may fail without root)")
        else:
            print(f"  sudo: requires password (trying {pm} without sudo — may fail without root)")

        if pm == "apt":
            update_cmd = (["sudo", "apt-get", "update", "-qq"] if can_sudo else ["apt-get", "update", "-qq"])
            install_cmd = (["sudo", "apt-get", "install", "-y"] if can_sudo else ["apt-get", "install", "-y"]) + packages_needed
        elif pm == "pacman":
            install_cmd = (["sudo", "pacman", "-S", "--noconfirm"] if can_sudo else ["pacman", "-S", "--noconfirm"]) + packages_needed
        elif pm in ("dnf", "yum"):
            install_cmd = (["sudo", pm, "install", "-y"] if can_sudo else [pm, "install", "-y"]) + packages_needed
        elif pm == "zypper":
            install_cmd = (["sudo", "zypper", "install", "-y"] if can_sudo else ["zypper", "install", "-y"]) + packages_needed
        elif pm == "apk":
            install_cmd = (["sudo", "apk", "add"] if can_sudo else ["apk", "add"]) + packages_needed
        else:
            print(f"  WARNING: unsupported pm '{pm}'. Install manually.")
            return
        if pm == "apt":
            try:
                run(update_cmd, check=False)
            except Exception as e:
                print(f"  WARNING: apt update failed: {e}")
        try:
            run(install_cmd, check=False)
        except Exception as e:
            print(f"  WARNING: install failed: {e}")
            if not can_sudo and os.geteuid() != 0:
                manual_cmd = " ".join(install_cmd)
                print(f"  No root/sudo access. Please run manually as root: {manual_cmd}")
    elif is_macos():
        if command_exists("brew"):
            run(["brew", "install"] + packages_needed, check=False)
        else:
            print(f"  WARNING: Homebrew not found. Please install manually: {', '.join(packages_needed)}")
    elif is_windows():
        if command_exists("winget"):
            for pkg in packages_needed:
                run(["winget", "install", "--accept-source-agreements", "--accept-package-agreements", pkg], check=False, shell=True)
        elif command_exists("choco"):
            run(["choco", "install", "-y"] + packages_needed, check=False, shell=True)
        else:
            print(f"  WARNING: winget/choco not found. Please install manually: {', '.join(packages_needed)}")


def install_ollama():
    print("\n[OLLAMA] Checking Ollama (used by VADAR chat in Project Eva)...")
    if command_exists("ollama"):
        print("  Ollama: already installed.")
        return
    print("  Installing Ollama...")
    if is_linux() or is_macos():
        try:
            subprocess.run("curl -fsSL https://ollama.com/install.sh | sh", shell=True, check=True)
            print("  Ollama installed successfully.")
        except Exception as e:
            print(f"  WARNING: Ollama installation failed: {e}")
            print("  Please install manually: curl -fsSL https://ollama.com/install.sh | sh")
    elif is_windows():
        try:
            subprocess.run("irm https://ollama.com/install.ps1 | iex", shell=True, check=True)
            print("  Ollama installed successfully.")
        except Exception as e:
            print(f"  WARNING: Ollama installation failed: {e}")
            print("  Please install manually: irm https://ollama.com/install.ps1 | iex")
    else:
        print("  WARNING: Unsupported OS for automatic Ollama installation.")


def setup_hf_token():
    token_file = os.path.join(SRC_DIR, "HF_TOKEN.txt")
    if not os.path.exists(token_file):
        with open(token_file, "w") as f:
            f.write("# Paste your HuggingFace token here\n")
            f.write("# Get your token from: https://huggingface.co/settings/tokens\n")
            f.write("# Required for: Flux 2 Dev (gated), MiniMax H3, SAM 3.1 (gated)\n")
        print("\n  HF_TOKEN.txt created. Paste your HuggingFace token in it for gated models.")
    else:
        with open(token_file, "r") as f:
            content = f.read().strip()
            lines = [line for line in content.split("\n") if line and not line.startswith("#")]
            if lines:
                print("  HF_TOKEN: found in HF_TOKEN.txt")
            else:
                print("  HF_TOKEN.txt exists but is empty. Paste your token for gated models.")


def venv_python_path(env_dir):
    if is_windows():
        return os.path.join(env_dir, "Scripts", "python.exe")
    return os.path.join(env_dir, "bin", "python")


def _venv_has_pip(py_bin):
    try:
        r = subprocess.run([py_bin, "-m", "pip", "--version"], capture_output=True, timeout=15,
                           env=_clean_subprocess_env())
        return r.returncode == 0
    except Exception:
        return False


def _clean_subprocess_env():
    env = os.environ.copy()
    for k in ("PYTHONPATH", "PYTHONSTARTUP", "PYTHONHOME"):
        env.pop(k, None)
    env["PYTHONNOUSERSITE"] = "1"
    env["PIP_NO_INPUT"] = "1"
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    return env


def _clean_partial_venv(env_dir):
    venv_artifacts = ("bin", "lib", "lib64", "include", "share", "pyvenv.cfg", "get-pip.py", ".venv")
    for name in venv_artifacts:
        path = os.path.join(env_dir, name)
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        elif os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass


def create_venv(env_name):
    env_dir = os.path.join(ENVS_DIR, env_name)
    py_bin = venv_python_path(env_dir)
    if os.path.exists(py_bin) and _venv_has_pip(py_bin):
        print(f"  venv already exists: {env_dir}")
        return py_bin
    if os.path.exists(env_dir):
        _clean_partial_venv(env_dir)
    os.makedirs(ENVS_DIR, exist_ok=True)
    print(f"  Creating venv: {env_dir}")
    try:
        subprocess.run([sys.executable, "-m", "venv", env_dir], check=True)
    except Exception:
        try:
            _clean_partial_venv(env_dir)
            subprocess.run([sys.executable, "-m", "venv", "--without-pip", env_dir], check=True)
        except Exception as e:
            print(f"  ERROR: venv creation failed: {e}")
            return None
        if not os.path.exists(py_bin):
            print(f"  ERROR: venv python not found at {py_bin}")
            return None
        print("  ensurepip not available, bootstrapping pip via get-pip.py...")
        try:
            import urllib.request
            get_pip_path = os.path.join(env_dir, "get-pip.py")
            urllib.request.urlretrieve("https://bootstrap.pypa.io/get-pip.py", get_pip_path)
            subprocess.run([py_bin, get_pip_path], check=True, env=_clean_subprocess_env())
            os.remove(get_pip_path)
        except Exception as e:
            print(f"  ERROR: pip bootstrap failed: {e}")
            return None
    if not os.path.exists(py_bin):
        print(f"  ERROR: venv python not found at {py_bin}")
        return None
    if not _venv_has_pip(py_bin):
        print(f"  ERROR: pip not functional in venv at {py_bin}")
        return None
    try:
        subprocess.run([py_bin, "-m", "pip", "install", "--upgrade", "pip"], check=False,
                       env=_clean_subprocess_env())
    except Exception:
        pass
    return py_bin


def install_venv_requirements(env_name, py_bin):
    req_file = os.path.join(ENVS_DIR, env_name, "requirements.txt")
    if not os.path.exists(req_file):
        print(f"  WARNING: requirements file not found: {req_file}")
        return False
    print(f"  Installing requirements for {env_name} (this may take a while)...")
    rc = subprocess.run([py_bin, "-m", "pip", "install", "-r", req_file],
                       env=_clean_subprocess_env()).returncode
    if rc != 0:
        print(f"  WARNING: pip install for {env_name} returned exit code {rc}")
        return False
    if not _post_install_extras(env_name, py_bin):
        return False
    return True


def _post_install_extras(env_name, py_bin):
    if env_name == "trellis":
        cuda_home = _ensure_cuda_dev_headers(py_bin)
        if cuda_home is None:
            return False
        if not _post_install_trellis_cuda_ext(py_bin, cuda_home):
            return False
        if not _post_install_flash_attn(py_bin, "trellis", cuda_home):
            return False
        return True
    if env_name == "animate":
        cuda_home = _ensure_cuda_dev_headers(py_bin)
        if cuda_home is None:
            return False
        if not _post_install_sam2(py_bin, cuda_home):
            return False
        return True
    return True


def _venv_site_packages(py_bin):
    try:
        r = subprocess.run([py_bin, "-c", "import site; print(site.getsitepackages()[0])"],
                           capture_output=True, text=True, timeout=15)
        if r.returncode == 0:
            return r.stdout.strip()
    except Exception:
        pass
    return None


def _ensure_cuda_dev_headers(py_bin):
    print("  Checking CUDA dev headers (cusparse.h, cublas_v2.h, curand.h, cufft.h, cusolver_common.h)...")
    system_cuda = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    system_include = os.path.join(system_cuda, "include")
    needed = ["cusparse.h", "cublas_v2.h", "curand.h", "cufft.h", "cusolver_common.h"]
    missing = [h for h in needed if not os.path.exists(os.path.join(system_include, h))]
    if not missing:
        print(f"  CUDA dev headers OK (system: {system_include}).")
        return system_cuda

    print(f"  Missing CUDA dev headers in {system_include}: {', '.join(missing)}")
    site_packages = _venv_site_packages(py_bin)
    if site_packages is None:
        print("  ERROR: cannot determine venv site-packages path.")
        return None

    nvidia_root = os.path.join(site_packages, "nvidia")
    if not os.path.isdir(nvidia_root):
        print(f"  ERROR: nvidia pip packages not found in {nvidia_root}.")
        print("  Make sure torch is installed in this venv (it bundles the nvidia-* packages).")
        return None

    can_write_system = False
    if os.path.isdir(system_include):
        try:
            test_file = os.path.join(system_include, ".voder_write_test")
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
            can_write_system = True
        except OSError:
            can_write_system = False

    if can_write_system:
        print(f"  Copying missing headers from nvidia pip packages into {system_include}...")
        copied = 0
        for pkg_dir in sorted(os.listdir(nvidia_root)):
            pkg_inc = os.path.join(nvidia_root, pkg_dir, "include")
            if not os.path.isdir(pkg_inc):
                continue
            for entry in os.listdir(pkg_inc):
                src = os.path.join(pkg_inc, entry)
                dst = os.path.join(system_include, entry)
                if not os.path.isfile(src):
                    continue
                if os.path.exists(dst):
                    continue
                try:
                    shutil.copy2(src, dst)
                    copied += 1
                except OSError:
                    pass
        print(f"  Copied {copied} header file(s) into {system_include}.")
        still_missing = [h for h in needed if not os.path.exists(os.path.join(system_include, h))]
        if not still_missing:
            print(f"  CUDA dev headers now present in {system_include}.")
            return system_cuda
        print(f"  ERROR: headers still missing after copy: {', '.join(still_missing)}")
        return None

    print(f"  Cannot write to {system_include} (not root). Building fallback synthetic CUDA_HOME...")
    synthetic_cuda = os.path.join(tempfile.gettempdir(), "voder_cuda_home")
    if os.path.exists(synthetic_cuda):
        shutil.rmtree(synthetic_cuda, ignore_errors=True)
    synth_include = os.path.join(synthetic_cuda, "include")
    synth_lib64 = os.path.join(synthetic_cuda, "lib64")
    synth_bin = os.path.join(synthetic_cuda, "bin")
    synth_nvvm_bin = os.path.join(synthetic_cuda, "nvvm", "bin")
    synth_nvvm_lib64 = os.path.join(synthetic_cuda, "nvvm", "lib64")
    synth_nvvm_libdevice = os.path.join(synthetic_cuda, "nvvm", "libdevice")
    synth_nvvm_include = os.path.join(synthetic_cuda, "nvvm", "include")
    for d in (synth_include, synth_lib64, synth_bin,
             synth_nvvm_bin, synth_nvvm_lib64, synth_nvvm_libdevice, synth_nvvm_include):
        os.makedirs(d, exist_ok=True)

    print(f"  Building synthetic CUDA_HOME at {synthetic_cuda}...")

    def _symlink_into(src_dir, dst_dir):
        if not os.path.isdir(src_dir):
            return
        for entry in os.listdir(src_dir):
            src = os.path.join(src_dir, entry)
            dst = os.path.join(dst_dir, entry)
            if os.path.lexists(dst):
                continue
            try:
                os.symlink(src, dst)
            except OSError:
                pass

    _symlink_into(system_include, synth_include)
    for pkg_dir in sorted(os.listdir(nvidia_root)):
        pkg_path = os.path.join(nvidia_root, pkg_dir)
        if not os.path.isdir(pkg_path):
            continue
        _symlink_into(os.path.join(pkg_path, "include"), synth_include)
        _symlink_into(os.path.join(pkg_path, "lib"), synth_lib64)
        _symlink_into(os.path.join(pkg_path, "lib64"), synth_lib64)

    system_bin = os.path.join(system_cuda, "bin")
    _symlink_into(system_bin, synth_bin)
    nvcc_pkg_bin = os.path.join(nvidia_root, "cuda_nvcc", "bin")
    _symlink_into(nvcc_pkg_bin, synth_bin)
    nvcc_pkg_nvvm = os.path.join(nvidia_root, "cuda_nvcc", "nvvm")
    if os.path.isdir(nvcc_pkg_nvvm):
        _symlink_into(os.path.join(nvcc_pkg_nvvm, "bin"), synth_nvvm_bin)
        _symlink_into(os.path.join(nvcc_pkg_nvvm, "lib64"), synth_nvvm_lib64)
        _symlink_into(os.path.join(nvcc_pkg_nvvm, "libdevice"), synth_nvvm_libdevice)
        _symlink_into(os.path.join(nvcc_pkg_nvvm, "include"), synth_nvvm_include)
    system_nvvm = os.path.join(system_cuda, "nvvm")
    if os.path.isdir(system_nvvm):
        _symlink_into(os.path.join(system_nvvm, "bin"), synth_nvvm_bin)
        _symlink_into(os.path.join(system_nvvm, "lib64"), synth_nvvm_lib64)
        _symlink_into(os.path.join(system_nvvm, "libdevice"), synth_nvvm_libdevice)
        _symlink_into(os.path.join(system_nvvm, "include"), synth_nvvm_include)

    still_missing = [h for h in needed if not os.path.exists(os.path.join(synth_include, h))]
    if still_missing:
        print(f"  ERROR: headers still missing: {', '.join(still_missing)}")
        return None

    print(f"  Synthetic CUDA_HOME ready: {synthetic_cuda}")
    return synthetic_cuda


def _post_install_flash_attn(py_bin, env_name, cuda_home):
    print(f"  Installing flash-attn for {env_name} (no build isolation)...")
    print("  This requires CUDA Toolkit (nvcc) matching the installed PyTorch's CUDA version.")
    env = _clean_subprocess_env()
    env["CUDA_HOME"] = cuda_home
    env["CUDA_PATH"] = cuda_home
    cuda_bin = os.path.join(cuda_home, "bin")
    if os.path.isdir(cuda_bin):
        env["PATH"] = cuda_bin + os.pathsep + env.get("PATH", "")
    rc = subprocess.run(
        [py_bin, "-m", "pip", "install", "flash-attn==2.7.3", "--no-build-isolation"],
        env=env
    ).returncode
    if rc != 0:
        print(f"  WARNING: flash-attn install failed for {env_name}.")
        print("  This usually means CUDA Toolkit (nvcc) version mismatch with PyTorch's CUDA.")
        print("  Install CUDA Toolkit matching your PyTorch's CUDA version, then re-run with --force.")
        print("  Or skip flash-attn — the model will still work, just slower (no flash attention kernel).")
        return False
    return True


def _post_install_sam2(py_bin, cuda_home):
    print("  Installing SAM-2 (no build isolation)...")
    print("  This requires CUDA Toolkit (nvcc) matching the installed PyTorch's CUDA version.")
    ext_dir = os.path.join(tempfile.gettempdir(), "sam2_install")
    if os.path.exists(ext_dir):
        shutil.rmtree(ext_dir, ignore_errors=True)
    os.makedirs(ext_dir, exist_ok=True)
    target = os.path.join(ext_dir, "sam2")
    print("    Cloning SAM-2...")
    rc = subprocess.run(
        ["git", "clone", "https://github.com/facebookresearch/sam2.git", target],
        env=_clean_subprocess_env()
    ).returncode
    if rc != 0:
        print("  WARNING: failed to clone SAM-2")
        return False
    print("    Installing SAM-2 (no build isolation)...")
    env = _clean_subprocess_env()
    env["CUDA_HOME"] = cuda_home
    env["CUDA_PATH"] = cuda_home
    cuda_bin = os.path.join(cuda_home, "bin")
    if os.path.isdir(cuda_bin):
        env["PATH"] = cuda_bin + os.pathsep + env.get("PATH", "")
    rc = subprocess.run(
        [py_bin, "-m", "pip", "install", "-e", target, "--no-build-isolation"],
        env=env
    ).returncode
    if rc != 0:
        print("  WARNING: SAM-2 install failed.")
        print("  This usually means CUDA Toolkit (nvcc) version mismatch with PyTorch's CUDA.")
        print("  Install CUDA Toolkit matching your PyTorch's CUDA version, then re-run with --force.")
        return False
    print("  SAM-2 installed.")
    return True


def _post_install_trellis_cuda_ext(py_bin, cuda_home):
    print("  Installing CUDA-compiled extensions for TRELLIS.2 (nvdiffrast, cumesh, o-voxel, flexgemm)...")
    print(f"  Using CUDA_HOME: {cuda_home}")
    ext_dir = os.path.join(tempfile.gettempdir(), "trellis2_extensions")
    if os.path.exists(ext_dir):
        shutil.rmtree(ext_dir, ignore_errors=True)
    os.makedirs(ext_dir, exist_ok=True)
    packages = [
        ("nvdiffrast", "https://github.com/NVlabs/nvdiffrast.git", "v0.4.0"),
        ("CuMesh", "https://github.com/JeffreyXiang/CuMesh.git", None),
        ("FlexGEMM", "https://github.com/JeffreyXiang/FlexGEMM.git", None),
    ]
    build_env = _clean_subprocess_env()
    build_env["CUDA_HOME"] = cuda_home
    build_env["CUDA_PATH"] = cuda_home
    cuda_bin = os.path.join(cuda_home, "bin")
    if os.path.isdir(cuda_bin):
        existing_path = build_env.get("PATH", "")
        build_env["PATH"] = cuda_bin + os.pathsep + existing_path
    for name, url, tag in packages:
        target = os.path.join(ext_dir, name)
        clone_cmd = ["git", "clone", "--recursive"]
        if tag:
            clone_cmd += ["-b", tag]
        clone_cmd += [url, target]
        print(f"    Cloning {name}...")
        rc = subprocess.run(clone_cmd, env=_clean_subprocess_env()).returncode
        if rc != 0:
            print(f"  WARNING: failed to clone {name} from {url}")
            return False
        print(f"    Installing {name} (no build isolation)...")
        rc = subprocess.run([py_bin, "-m", "pip", "install", target, "--no-build-isolation"],
                           env=build_env).returncode
        if rc != 0:
            print(f"  WARNING: failed to install {name}")
            print(f"  This usually means CUDA Toolkit (nvcc) is not installed or not on PATH.")
            print(f"  Install CUDA Toolkit 12.4+ from https://developer.nvidia.com/cuda-toolkit-archive")
            print(f"  Then re-run: python setup.py --envs trellis --force")
            return False
    ovoxel_target = _clone_trellis_o_voxel(ext_dir)
    if ovoxel_target is None:
        return False
    print(f"    Installing o-voxel (no build isolation)...")
    rc = subprocess.run([py_bin, "-m", "pip", "install", ovoxel_target, "--no-build-isolation"],
                       env=build_env).returncode
    if rc != 0:
        print(f"  WARNING: failed to install o-voxel")
        return False
    print("  CUDA extensions installed.")
    return True


def _clone_trellis_o_voxel(ext_dir):
    target = os.path.join(ext_dir, "TRELLIS.2")
    if os.path.exists(target):
        shutil.rmtree(target, ignore_errors=True)
    print("    Cloning TRELLIS.2 (for o-voxel subdirectory, with submodules for Eigen)...")
    rc = subprocess.run(
        ["git", "clone", "--recursive", "https://github.com/microsoft/TRELLIS.2.git", target],
        env=_clean_subprocess_env()
    ).returncode
    if rc != 0:
        print("  WARNING: failed to clone TRELLIS.2 for o-voxel")
        return None
    subprocess.run(["git", "submodule", "update", "--init", "--recursive"],
                   cwd=target, env=_clean_subprocess_env())
    ovoxel_path = os.path.join(target, "o-voxel")
    if not os.path.isdir(ovoxel_path):
        print(f"  WARNING: o-voxel subdirectory not found in TRELLIS.2 clone at {ovoxel_path}")
        return None
    eigen_path = os.path.join(ovoxel_path, "third_party", "eigen")
    if not os.path.isdir(eigen_path):
        print(f"  WARNING: Eigen submodule not found at {eigen_path}")
        print("  o-voxel requires Eigen headers (git submodule).")
        return None
    print(f"    Eigen headers found at {eigen_path}")
    return ovoxel_path


def setup_eva_env(env_name, force=False):
    print(f"\n[EVA-ENV] {env_name} — {EVA_ENVS.get(env_name, '?')}")
    env_dir = os.path.join(ENVS_DIR, env_name)
    py_bin = venv_python_path(env_dir)
    if os.path.exists(py_bin) and _venv_has_pip(py_bin) and not force:
        print(f"  Already set up: {env_dir}")
        print(f"  To reinstall, run: python setup.py --envs {env_name} --force")
        return True
    if force and os.path.exists(env_dir):
        print(f"  Removing existing venv (force)...")
        _clean_partial_venv(env_dir)
    py_bin = create_venv(env_name)
    if py_bin is None:
        return False
    ok = install_venv_requirements(env_name, py_bin)
    if not ok:
        print(f"  Failed to install requirements for {env_name}.")
        print(f"  You can retry with: {py_bin} -m pip install -r {os.path.join(env_dir, 'requirements.txt')}")
        return False
    print(f"  [OK] {env_name} env ready: {env_dir}")
    return True


def setup_all_eva_envs(force=False):
    print("\n[PROJECT EVA] Setting up isolated environments for each Eva model...")
    os.makedirs(ENVS_DIR, exist_ok=True)
    results = {}
    for name in EVA_ENVS:
        results[name] = setup_eva_env(name, force=force)
    print("\n[EVA-ENV] Summary:")
    for name, ok in results.items():
        mark = "OK" if ok else "--"
        print(f"  [{mark}] {name:8s} — {EVA_ENVS[name]}")
    return results


def verify_installation():
    print("\n[VERIFY] Installation status:")
    if command_exists("ffmpeg"):
        print("  ffmpeg: OK")
    else:
        print("  ffmpeg: NOT FOUND")
    if command_exists("sox"):
        print("  sox: OK")
    else:
        print("  sox: NOT FOUND")
    if command_exists("ollama"):
        print("  ollama: OK")
    else:
        print("  ollama: NOT FOUND (VADAR chat will not work)")
    if command_exists("git"):
        print("  git: OK")
    else:
        print("  git: NOT FOUND")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  PyTorch CUDA: {torch.cuda.get_device_name(0)}")
        else:
            print("  PyTorch CUDA: not available (CPU mode)")
    except ImportError:
        print("  PyTorch: not found")
    print("\n  Eva model envs:")
    for name in EVA_ENVS:
        env_dir = os.path.join(ENVS_DIR, name)
        py_bin = venv_python_path(env_dir)
        ok = os.path.exists(py_bin) and _venv_has_pip(py_bin)
        mark = "OK" if ok else "--"
        print(f"    [{mark}] {name:8s} — {EVA_ENVS[name]}")


def main():
    parser = argparse.ArgumentParser(description="VODER Setup")
    parser.add_argument("--envs", nargs="*", default=None,
                        help=f"Set up specific Eva model envs (choices: {list(EVA_ENVS.keys())}, or 'all').")
    parser.add_argument("--force", action="store_true", help="Force re-create envs even if they exist.")
    parser.add_argument("--skip-main", action="store_true", help="Skip main VODER requirements install.")
    parser.add_argument("--skip-system", action="store_true", help="Skip system package install.")
    parser.add_argument("--skip-ollama", action="store_true", help="Skip Ollama install.")
    args = parser.parse_args()

    print("""
============================================================
  VODER Setup
============================================================
""")

    envs_only = args.envs is not None

    if not envs_only:
        if not args.skip_system:
            print("[1/5] Installing system packages (ffmpeg, sox, git, ...)...")
            try_install_system_packages()
        else:
            print("[1/5] Skipping system packages.")

        if not args.skip_ollama:
            print("\n[2/5] Installing Ollama (used by VADAR chat)...")
            install_ollama()
        else:
            print("[2/5] Skipping Ollama.")

        if not args.skip_main:
            print("\n[3/5] Installing VODER Python requirements...")
            run([sys.executable, "-m", "pip", "install", "-r", REQUIREMENTS_FILE])
            print("\n  Installing protobuf 5.29.6 (fix for descript-audiotools)...")
            run([sys.executable, "-m", "pip", "install", "--upgrade", "protobuf==5.29.6"])
        else:
            print("[3/5] Skipping VODER requirements.")

        print("\n[4/5] Setting up Project Eva model environments...")
        setup_all_eva_envs(force=args.force)

        setup_hf_token()

        print("\n[5/5] Verifying installation...")
        verify_installation()
    else:
        envs = args.envs
        if not envs or envs == ["all"] or envs == []:
            envs = list(EVA_ENVS.keys())
        print("[Eva envs only] Setting up: " + ", ".join(envs))
        for name in envs:
            if name not in EVA_ENVS:
                print(f"  Unknown env '{name}'. Available: {list(EVA_ENVS.keys())}")
                continue
            setup_eva_env(name, force=args.force)
        print("\n[EVA-ENV] Status:")
        for name in EVA_ENVS:
            env_dir = os.path.join(ENVS_DIR, name)
            py_bin = venv_python_path(env_dir)
            ok = os.path.exists(py_bin) and _venv_has_pip(py_bin)
            mark = "OK" if ok else "--"
            print(f"  [{mark}] {name:8s} — {EVA_ENVS[name]}")

    print("""
============================================================
  VODER Setup Complete
============================================================

  Quick start:
    python voder.py tts "Hello world" voice:"narrator"
    python voder.py stt "audio.wav" timestamp dialogue
    python voder.py ttm lyrics "Walking down the street" styling "upbeat pop" 30
    python voder.py sts base "input.wav" target "voice.wav"
    python voder.py sfx sound "thunder rumbling" duration 10
    python voder.py cli
    python voder.py gui

  Project Eva DLC (image, video, world, chat models):
    python voder.py eva tti gen "a cyberpunk city at night"
    python voder.py eva tti mini gen "a cyberpunk city at night"
    python voder.py eva tti edit "input.png" desc "add a red sky"
    python voder.py eva ttv gen "a cat playing piano" duration 10
    python voder.py eva ttv animify "character.png" reference "pose.mp4"
    python voder.py eva ttv edit "input.mp4" desc "make it night time"
    python voder.py eva ttv lipsync "face.png" reference "voice.wav"
    python voder.py eva ttt gen "how are you?"
    python voder.py eva ttw gen "a medieval castle on a hill"
    python voder.py eva ttw objectify "character.png"
    python voder.py eva ttw edit objectify "character.glb" reference "bronze_texture.png"

  Each Eva model runs in its own venv under src/envs/<model>/.
  Set them up separately if you skipped them in the main install:
    python setup.py --envs all
    python setup.py --envs flux2
    python setup.py --envs flux2 h3 --force

  Klarify DLC (upscale, enhance, interpolate):
    python voder.py klarify upscale "image.png"
    python voder.py klarify enhance "video.mp4"
    python voder.py klarify interpolate "video.mp4"

  See docs/Guide.md for the full guide.
============================================================
""")


if __name__ == "__main__":
    main()
