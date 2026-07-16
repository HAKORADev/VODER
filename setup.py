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
    if not command_exists("zstd"):
        packages_needed.append("zstd")
    if is_linux() and not command_exists("lspci") and not command_exists("lshw"):
        packages_needed.append("lshw")

    packages_needed = list(dict.fromkeys(packages_needed))

    if not packages_needed:
        print("  All system packages already installed.")
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


def main():
    print("""
============================================================
  VODER Setup
============================================================
""")

    print("[1/4] Installing system packages (ffmpeg, sox)...")
    install_system_packages()

    print("\n[2/4] Installing Python requirements...")
    run([sys.executable, "-m", "pip", "install", "-r", REQUIREMENTS_FILE])

    print("\n[3/4] Installing protobuf 5.29.6 (fix for descript-audiotools)...")
    run([sys.executable, "-m", "pip", "install", "--upgrade", "protobuf==5.29.6"])

    print("\n[4/4] Verifying installation...")

    if command_exists("ffmpeg"):
        print("  ffmpeg: OK")
    else:
        print("  ffmpeg: NOT FOUND")

    if command_exists("sox"):
        print("  sox: OK")
    else:
        print("  sox: NOT FOUND")

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
    python voder.py tts "Hello world" voice:"narrator"
    python voder.py quest download "https://youtube.com/watch?v=..."
    python voder.py quest media-search youtube "lofi beats" 10
    python voder.py cli

  See docs/Guide.md for the full guide.
============================================================
""")


if __name__ == "__main__":
    main()
