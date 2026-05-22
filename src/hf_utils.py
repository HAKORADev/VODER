"""
HuggingFace Model Utilities for VODER

This module provides utilities for downloading models from HuggingFace Hub.
All models are stored in the centralized models directory: src/models/checkpoints/

NOTE: For Seed-VC models, use the centralized paths from voder.py:
  - SEED_VC_V1_DIR for Seed-VC v1 checkpoints
  - SEED_VC_V2_DIR for Seed-VC v2 checkpoints
"""
import os
from huggingface_hub import hf_hub_download

# Get the src directory (parent of this file's directory)
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_MODELS_DIR = os.path.join(_SRC_DIR, "models")
_CHECKPOINTS_DIR = os.path.join(_MODELS_DIR, "checkpoints")


def load_custom_model_from_hf(repo_id, model_filename="pytorch_model.bin", config_filename=None, target_dir=None):
    """
    Download model files from HuggingFace Hub.
    
    Args:
        repo_id: HuggingFace repository ID
        model_filename: Name of the model file to download
        config_filename: Optional config file to download
        target_dir: Target directory (defaults to models/checkpoints/)
    
    Returns:
        Path to the downloaded model file, or tuple of (model_path, config_path) if config_filename is provided
    """
    if target_dir is None:
        target_dir = _CHECKPOINTS_DIR
    os.makedirs(target_dir, exist_ok=True)
    model_path = hf_hub_download(repo_id=repo_id, filename=model_filename, cache_dir=target_dir)
    if config_filename is None:
        return model_path
    config_path = hf_hub_download(repo_id=repo_id, filename=config_filename, cache_dir=target_dir)
    return model_path, config_path
