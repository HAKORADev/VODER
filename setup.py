#!/usr/bin/env python3

import subprocess
import sys
import os
import platform

REQUIREMENTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements.txt")


def run(cmd, env=None, check=True, shell=False):
    print(f"\n{'='*60}")
    print(f"  Running: {cmd if isinstance(cmd, str) else ' '.join(cmd)}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd, env=env, shell=shell)
    if check and result.returncode != 0:
        print(f"\n  WARNING: Command returned exit code {result.returncode}")
    return result.returncode


def is_linux():
    return platform.system() == "Linux"


def is_windows():
    return platform.system() == "Windows"


def is_macos():
    return platform.system() == "Darwin"


def package_manager():
    if os.path.exists("/usr/bin/apt-get") or os.path.exists("/usr/bin/apt"):
        return "apt"
    if os.path.exists("/usr/bin/pacman"):
        return "pacman"
    if os.path.exists("/usr/bin/dnf"):
        return "dnf"
    if os.path.exists("/usr/bin/yum"):
        return "yum"
    if os.path.exists("/usr/bin/zypper"):
        return "zypper"
    return None


def command_exists(cmd):
    try:
        subprocess.run(["which", cmd], capture_output=True, check=True)
        return True
    except Exception:
        try:
            subprocess.run(["where", cmd], capture_output=True, check=True, shell=True)
            return True
        except Exception:
            return False


def install_system_packages():
    print("\n[SYSTEM] Checking system packages...")

    packages_needed = []
    if not command_exists("ffmpeg"):
        packages_needed.append("ffmpeg")
    if not command_exists("sox"):
        packages_needed.append("sox")
    if not command_exists("soxi"):
        packages_needed.append("sox")

    if not packages_needed:
        print("  ffmpeg, sox: already installed.")
        return

    if is_linux():
        pm = package_manager()
        if pm == "apt":
            run(["sudo", "apt-get", "update", "-qq"], check=False)
            run(["sudo", "apt-get", "install", "-y"] + packages_needed, check=False)
        elif pm == "pacman":
            run(["sudo", "pacman", "-S", "--noconfirm"] + packages_needed, check=False)
        elif pm in ("dnf", "yum"):
            run(["sudo", pm, "install", "-y"] + packages_needed, check=False)
        elif pm == "zypper":
            run(["sudo", "zypper", "install", "-y"] + packages_needed, check=False)
        else:
            print(f"  WARNING: Unknown package manager. Please install manually: {', '.join(packages_needed)}")
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
    print("\n[OLLAMA] Checking Ollama...")

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


def ensure_ollama_running():
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("  Ollama service: running.")
            return True
    except Exception:
        pass

    print("  Starting Ollama service...")
    try:
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        import time
        time.sleep(3)
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("  Ollama service: started.")
            return True
    except Exception as e:
        print(f"  WARNING: Could not start Ollama service: {e}")
    return False


def main():
    print("""
============================================================
  VODER Setup
============================================================
""")

    print("[1/5] Installing system packages (ffmpeg, sox)...")
    install_system_packages()

    print("\n[2/5] Installing Ollama...")
    install_ollama()

    print("\n[3/5] Installing Python requirements...")
    run([sys.executable, "-m", "pip", "install", "-r", REQUIREMENTS_FILE])

    print("\n[4/5] Installing protobuf 5.29.6 (fix for descript-audiotools)...")
    run([sys.executable, "-m", "pip", "install", "--upgrade", "protobuf==5.29.6"])

    print("\n[5/5] Verifying installation...")

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
        ensure_ollama_running()
    else:
        print("  ollama: NOT FOUND (VADAR Lite will not work)")

    try:
        import ollama
        print("  Python ollama library: OK")
    except ImportError:
        print("  Python ollama library: NOT FOUND")

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
