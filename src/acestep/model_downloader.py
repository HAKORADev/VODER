"""
ACE-Step Model Downloader

This module provides functionality to download models from HuggingFace Hub.
It supports automatic downloading when models are not found locally,
as well as a CLI for manual downloads.
"""

import os
import sys
import argparse
from typing import Optional, List, Dict, Tuple
from pathlib import Path

from huggingface_hub import snapshot_download, HfApi
from loguru import logger


# Model registry: maps local directory names to HuggingFace repo IDs
# Main model contains core components (vae, text_encoder, default DiT)
MAIN_MODEL_REPO = "ACE-Step/Ace-Step1.5"

# Sub-models that can be downloaded separately into the checkpoints directory
SUBMODEL_REGISTRY: Dict[str, str] = {
    # LM models
    "acestep-5Hz-lm-0.6B": "ACE-Step/acestep-5Hz-lm-0.6B",
    "acestep-5Hz-lm-4B": "ACE-Step/acestep-5Hz-lm-4B",
    # DiT models
    "acestep-v15-turbo-shift3": "ACE-Step/acestep-v15-turbo-shift3",
    "acestep-v15-sft": "ACE-Step/acestep-v15-sft",
    "acestep-v15-base": "ACE-Step/acestep-v15-base",
    "acestep-v15-turbo-shift1": "ACE-Step/acestep-v15-turbo-shift1",
    "acestep-v15-turbo-continuous": "ACE-Step/acestep-v15-turbo-continuous",
}

# Components that come from the main model repo (ACE-Step/Ace-Step1.5)
MAIN_MODEL_COMPONENTS = [
    "acestep-v15-turbo",      # Default DiT model
    "vae",                     # VAE for audio encoding/decoding
    "Qwen3-Embedding-0.6B",    # Text encoder
    "acestep-5Hz-lm-1.7B",     # Default LM model (1.7B)
]

# Default LM model (included in main model)
DEFAULT_LM_MODEL = "acestep-5Hz-lm-1.7B"


def get_project_root() -> Path:
    """Get the project root directory."""
    current_file = Path(__file__).resolve()
    return current_file.parent.parent


def get_checkpoints_dir(custom_dir: Optional[str] = None) -> Path:
    """Get the checkpoints directory path."""
    if custom_dir:
        return Path(custom_dir)
    return get_project_root() / "checkpoints"


def check_main_model_exists(checkpoints_dir: Optional[Path] = None) -> bool:
    """
    Check if the main model components exist in the checkpoints directory.
    
    Returns:
        True if all main model components exist, False otherwise.
    """
    if checkpoints_dir is None:
        checkpoints_dir = get_checkpoints_dir()
    
    for component in MAIN_MODEL_COMPONENTS:
        component_path = checkpoints_dir / component
        if not component_path.exists():
            return False
    return True


def check_model_exists(model_name: str, checkpoints_dir: Optional[Path] = None) -> bool:
    """
    Check if a specific model exists in the checkpoints directory.
    
    Args:
        model_name: Name of the model to check
        checkpoints_dir: Custom checkpoints directory (optional)
    
    Returns:
        True if the model exists, False otherwise.
    """
    if checkpoints_dir is None:
        checkpoints_dir = get_checkpoints_dir()
    
    model_path = checkpoints_dir / model_name
    return model_path.exists()


def list_available_models() -> Dict[str, str]:
    """
    List all available models for download.
    
    Returns:
        Dictionary mapping local names to HuggingFace repo IDs.
    """
    models = {
        "main": MAIN_MODEL_REPO,
        **SUBMODEL_REGISTRY
    }
    return models


def download_main_model(
    checkpoints_dir: Optional[Path] = None,
    force: bool = False,
    token: Optional[str] = None,
) -> Tuple[bool, str]:
    """
    Download the main ACE-Step model from HuggingFace.
    
    The main model includes:
    - acestep-v15-turbo (default DiT model)
    - vae (audio encoder/decoder)
    - Qwen3-Embedding-0.6B (text encoder)
    - acestep-5Hz-lm-1.7B (default LM model)
    
    Args:
        checkpoints_dir: Custom checkpoints directory (optional)
        force: Force re-download even if model exists
        token: HuggingFace token for private repos (optional)
    
    Returns:
        Tuple of (success, message)
    """
    if checkpoints_dir is None:
        checkpoints_dir = get_checkpoints_dir()
    
    # Ensure checkpoints directory exists
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    if not force and check_main_model_exists(checkpoints_dir):
        return True, f"Main model already exists at {checkpoints_dir}"
    
    try:
        print(f"Downloading main model from {MAIN_MODEL_REPO}...")
        print(f"Destination: {checkpoints_dir}")
        print("This may take a while depending on your internet connection...")
        
        # Download the main model
        snapshot_download(
            repo_id=MAIN_MODEL_REPO,
            local_dir=str(checkpoints_dir),
            local_dir_use_symlinks=False,
            token=token,
        )
        
        return True, f"Successfully downloaded main model to {checkpoints_dir}"
    except Exception as e:
        error_msg = f"Failed to download main model: {str(e)}"
        logger.error(error_msg)
        return False, error_msg


def download_submodel(
    model_name: str,
    checkpoints_dir: Optional[Path] = None,
    force: bool = False,
    token: Optional[str] = None,
) -> Tuple[bool, str]:
    """
    Download a specific sub-model from HuggingFace.
    
    Args:
        model_name: Name of the model to download (must be in SUBMODEL_REGISTRY)
        checkpoints_dir: Custom checkpoints directory (optional)
        force: Force re-download even if model exists
        token: HuggingFace token for private repos (optional)
    
    Returns:
        Tuple of (success, message)
    """
    if model_name not in SUBMODEL_REGISTRY:
        available = ", ".join(SUBMODEL_REGISTRY.keys())
        return False, f"Unknown model '{model_name}'. Available models: {available}"
    
    if checkpoints_dir is None:
        checkpoints_dir = get_checkpoints_dir()
    
    # Ensure checkpoints directory exists
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = checkpoints_dir / model_name
    
    if not force and model_path.exists():
        return True, f"Model '{model_name}' already exists at {model_path}"
    
    repo_id = SUBMODEL_REGISTRY[model_name]
    
    try:
        print(f"Downloading {model_name} from {repo_id}...")
        print(f"Destination: {model_path}")
        
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(model_path),
            local_dir_use_symlinks=False,
            token=token,
        )
        
        return True, f"Successfully downloaded {model_name} to {model_path}"
    except Exception as e:
        error_msg = f"Failed to download {model_name}: {str(e)}"
        logger.error(error_msg)
        return False, error_msg


def download_all_models(
    checkpoints_dir: Optional[Path] = None,
    force: bool = False,
    token: Optional[str] = None,
) -> Tuple[bool, List[str]]:
    """
    Download all available models.
    
    Args:
        checkpoints_dir: Custom checkpoints directory (optional)
        force: Force re-download even if models exist
        token: HuggingFace token for private repos (optional)
    
    Returns:
        Tuple of (all_success, list of messages)
    """
    if checkpoints_dir is None:
        checkpoints_dir = get_checkpoints_dir()
    
    messages = []
    all_success = True
    
    # Download main model first
    success, msg = download_main_model(checkpoints_dir, force, token)
    messages.append(msg)
    if not success:
        all_success = False
    
    # Download all sub-models
    for model_name in SUBMODEL_REGISTRY:
        success, msg = download_submodel(model_name, checkpoints_dir, force, token)
        messages.append(msg)
        if not success:
            all_success = False
    
    return all_success, messages


def ensure_main_model(
    checkpoints_dir: Optional[Path] = None,
    token: Optional[str] = None,
) -> Tuple[bool, str]:
    """
    Ensure the main model is available, downloading if necessary.
    
    This function is designed to be called during initialization.
    It will only download if the model doesn't exist.
    
    Args:
        checkpoints_dir: Custom checkpoints directory (optional)
        token: HuggingFace token for private repos (optional)
    
    Returns:
        Tuple of (success, message)
    """
    if checkpoints_dir is None:
        checkpoints_dir = get_checkpoints_dir()
    
    if check_main_model_exists(checkpoints_dir):
        return True, "Main model is available"
    
    print("\n" + "=" * 60)
    print("Main model not found. Starting automatic download...")
    print("=" * 60 + "\n")
    
    return download_main_model(checkpoints_dir, token=token)


def ensure_lm_model(
    model_name: Optional[str] = None,
    checkpoints_dir: Optional[Path] = None,
    token: Optional[str] = None,
) -> Tuple[bool, str]:
    """
    Ensure an LM model is available, downloading if necessary.
    
    Args:
        model_name: Name of the LM model (defaults to DEFAULT_LM_MODEL)
        checkpoints_dir: Custom checkpoints directory (optional)
        token: HuggingFace token for private repos (optional)
    
    Returns:
        Tuple of (success, message)
    """
    if model_name is None:
        model_name = DEFAULT_LM_MODEL
    
    if checkpoints_dir is None:
        checkpoints_dir = get_checkpoints_dir()
    
    if check_model_exists(model_name, checkpoints_dir):
        return True, f"LM model '{model_name}' is available"
    
    # Check if this is a known LM model
    if model_name not in SUBMODEL_REGISTRY:
        # Check if it might be a variant name
        for known_model in SUBMODEL_REGISTRY:
            if "lm" in known_model.lower() and model_name.lower() in known_model.lower():
                model_name = known_model
                break
        else:
            return False, f"Unknown LM model: {model_name}"
    
    print("\n" + "=" * 60)
    print(f"LM model '{model_name}' not found. Starting automatic download...")
    print("=" * 60 + "\n")
    
    return download_submodel(model_name, checkpoints_dir, token=token)


def ensure_dit_model(
    model_name: str,
    checkpoints_dir: Optional[Path] = None,
    token: Optional[str] = None,
) -> Tuple[bool, str]:
    """
    Ensure a DiT model is available, downloading if necessary.
    
    Args:
        model_name: Name of the DiT model
        checkpoints_dir: Custom checkpoints directory (optional)
        token: HuggingFace token for private repos (optional)
    
    Returns:
        Tuple of (success, message)
    """
    if checkpoints_dir is None:
        checkpoints_dir = get_checkpoints_dir()
    
    if check_model_exists(model_name, checkpoints_dir):
        return True, f"DiT model '{model_name}' is available"
    
    # Check if this is the default turbo model (part of main)
    if model_name == "acestep-v15-turbo":
        return ensure_main_model(checkpoints_dir, token)
    
    # Check if it's a known sub-model
    if model_name in SUBMODEL_REGISTRY:
        print("\n" + "=" * 60)
        print(f"DiT model '{model_name}' not found. Starting automatic download...")
        print("=" * 60 + "\n")
        return download_submodel(model_name, checkpoints_dir, token=token)
    
    return False, f"Unknown DiT model: {model_name}"


def print_model_list():
    """Print formatted list of available models."""
    print("\nAvailable Models for Download:")
    print("=" * 60)
    
    print("\n[Main Model]")
    print(f"  main -> {MAIN_MODEL_REPO}")
    print("  Contains: vae, Qwen3-Embedding-0.6B, acestep-v15-turbo, acestep-5Hz-lm-1.7B")
    
    print("\n[Optional LM Models]")
    for name, repo in SUBMODEL_REGISTRY.items():
        if "lm" in name.lower():
            print(f"  {name} -> {repo}")
    
    print("\n[Optional DiT Models]")
    for name, repo in SUBMODEL_REGISTRY.items():
        if "lm" not in name.lower():
            print(f"  {name} -> {repo}")
    
    print("\n" + "=" * 60)


def main():
    """CLI entry point for model downloading."""
    parser = argparse.ArgumentParser(
        description="Download ACE-Step models from HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  acestep-download                          # Download main model (includes LM 1.7B)
  acestep-download --all                    # Download all available models
  acestep-download --model acestep-v15-sft  # Download a specific model
  acestep-download --list                   # List all available models

Alternative using huggingface-cli:
  huggingface-cli download ACE-Step/Ace-Step1.5 --local-dir ./checkpoints
  huggingface-cli download ACE-Step/acestep-5Hz-lm-0.6B --local-dir ./checkpoints/acestep-5Hz-lm-0.6B
        """
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        help="Specific model to download (use --list to see available models)"
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Download all available models"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all available models"
    )
    parser.add_argument(
        "--dir", "-d",
        type=str,
        default=None,
        help="Custom checkpoints directory (default: ./checkpoints)"
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force re-download even if model exists"
    )
    parser.add_argument(
        "--token", "-t",
        type=str,
        default=None,
        help="HuggingFace token for private repos"
    )
    parser.add_argument(
        "--skip-main",
        action="store_true",
        help="Skip downloading the main model (only download specified sub-model)"
    )
    
    args = parser.parse_args()
    
    # Handle --list
    if args.list:
        print_model_list()
        return 0
    
    # Get checkpoints directory
    checkpoints_dir = get_checkpoints_dir(args.dir) if args.dir else get_checkpoints_dir()
    print(f"Checkpoints directory: {checkpoints_dir}")
    
    # Handle --all
    if args.all:
        success, messages = download_all_models(checkpoints_dir, args.force, args.token)
        for msg in messages:
            print(msg)
        return 0 if success else 1
    
    # Handle --model
    if args.model:
        if args.model == "main":
            success, msg = download_main_model(checkpoints_dir, args.force, args.token)
        elif args.model in SUBMODEL_REGISTRY:
            # Download main model first if needed (unless --skip-main)
            if not args.skip_main and not check_main_model_exists(checkpoints_dir):
                print("Main model not found. Downloading main model first...")
                main_success, main_msg = download_main_model(checkpoints_dir, args.force, args.token)
                print(main_msg)
                if not main_success:
                    return 1
            
            success, msg = download_submodel(args.model, checkpoints_dir, args.force, args.token)
        else:
            print(f"Unknown model: {args.model}")
            print("Use --list to see available models")
            return 1
        
        print(msg)
        return 0 if success else 1
    
    # Default: download main model (includes default LM 1.7B)
    print("Downloading main model (includes vae, text encoder, DiT, and LM 1.7B)...")
    
    # Download main model
    success, msg = download_main_model(checkpoints_dir, args.force, args.token)
    print(msg)
    
    if success:
        print("\nDownload complete!")
        print(f"Models are available at: {checkpoints_dir}")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
