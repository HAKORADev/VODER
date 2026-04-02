"""
Training Configuration Classes

Contains dataclasses for LoRA and training configurations.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class LoRAConfig:
    """Configuration for LoRA (Low-Rank Adaptation) training.
    
    Attributes:
        r: LoRA rank (dimension of low-rank matrices)
        alpha: LoRA scaling factor (alpha/r determines the scaling)
        dropout: Dropout probability for LoRA layers
        target_modules: List of module names to apply LoRA to
        bias: Whether to train bias parameters ("none", "all", or "lora_only")
    """
    r: int = 8
    alpha: int = 16
    dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj"
    ])
    bias: str = "none"
    
    def to_dict(self):
        """Convert to dictionary for PEFT config."""
        return {
            "r": self.r,
            "lora_alpha": self.alpha,
            "lora_dropout": self.dropout,
            "target_modules": self.target_modules,
            "bias": self.bias,
        }


@dataclass
class TrainingConfig:
    """Configuration for LoRA training process.
    
    Training uses:
    - BFloat16 precision (only supported precision)
    - Discrete timesteps from turbo shift=3.0 schedule (8 steps)
    - Randomly samples one of 8 timesteps per training step:
      [1.0, 0.9545, 0.9, 0.8333, 0.75, 0.6429, 0.5, 0.3]
    
    Attributes:
        shift: Timestep shift factor (fixed at 3.0 for turbo model)
        num_inference_steps: Number of inference steps (fixed at 8 for turbo)
        learning_rate: Initial learning rate
        batch_size: Training batch size
        gradient_accumulation_steps: Number of gradient accumulation steps
        max_epochs: Maximum number of training epochs
        save_every_n_epochs: Save checkpoint every N epochs
        warmup_steps: Number of warmup steps for learning rate scheduler
        weight_decay: Weight decay for optimizer
        max_grad_norm: Maximum gradient norm for clipping
        mixed_precision: Always "bf16" (only supported precision)
        seed: Random seed for reproducibility
        output_dir: Directory to save checkpoints and logs
    """
    # Fixed for turbo model
    shift: float = 3.0  # Fixed: turbo uses shift=3.0
    num_inference_steps: int = 8  # Fixed: turbo uses 8 steps
    learning_rate: float = 1e-4
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    max_epochs: int = 100
    save_every_n_epochs: int = 10
    warmup_steps: int = 100
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    mixed_precision: str = "bf16"  # Fixed: only bf16 supported
    seed: int = 42
    output_dir: str = "./lora_output"
    
    # Data loading
    num_workers: int = 4
    pin_memory: bool = True
    
    # Logging
    log_every_n_steps: int = 10
    
    def to_dict(self):
        """Convert to dictionary."""
        return {
            "shift": self.shift,
            "num_inference_steps": self.num_inference_steps,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "max_epochs": self.max_epochs,
            "save_every_n_epochs": self.save_every_n_epochs,
            "warmup_steps": self.warmup_steps,
            "weight_decay": self.weight_decay,
            "max_grad_norm": self.max_grad_norm,
            "mixed_precision": self.mixed_precision,
            "seed": self.seed,
            "output_dir": self.output_dir,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "log_every_n_steps": self.log_every_n_steps,
        }
