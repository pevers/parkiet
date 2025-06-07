#!/usr/bin/env python3
"""
Configuration file for Whisper fine-tuning
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingConfig:
    """Training configuration parameters"""

    # Model settings
    model_name: str = "openai/whisper-small"
    language: str = "dutch"
    task: str = "transcribe"
    freeze_feature_encoder: bool = False
    freeze_encoder: bool = False

    # Data settings
    dataset_path: str = "../data/training"
    dataset_file: str = "whisper_dataset.json"
    max_duration_seconds: float = 30.0
    min_duration_seconds: float = 0.5
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None
    preprocessed_cache_dir: Optional[str] = "../data/training/.cache/preprocessed"
    dataset_seed: int = 1337

    # Training hyperparameters
    output_dir: str = "./whisper-small-dutch-cgn"
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-5
    warmup_steps: int = 500
    max_steps: int = 15000

    # Training settings
    gradient_checkpointing: bool = True
    fp16: bool = True
    dataloader_num_workers: int = 2

    # Evaluation and logging
    eval_strategy: str = "steps"
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 25
    save_total_limit: int = 3

    # Generation settings
    generation_max_length: int = 225

    # Monitoring
    report_to: list = None
    push_to_hub: bool = False
    run_name: Optional[str] = None  # Custom name for TensorBoard runs

    def __post_init__(self):
        if self.report_to is None:
            self.report_to = ["tensorboard"]


# Create default config instance
DEFAULT_CONFIG = TrainingConfig()


# Alternative configs for different scenarios
@dataclass
class QuickTestConfig(TrainingConfig):
    """Quick test configuration with smaller parameters"""

    max_steps: int = 2000
    eval_steps: int = 500
    save_steps: int = 500
    max_train_samples: int = 1000
    max_eval_samples: int = 100
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    run_name: str = "whisper-dutch-quick-test"


@dataclass
class ProductionConfig(TrainingConfig):
    """Production configuration with optimized parameters"""

    max_steps: int = 4500  # ~3 epochs for 90k samples (effective batch size 64)
    eval_steps: int = 750  # Evaluate ~6 times per epoch
    save_steps: int = 750  # Save ~6 times per epoch
    learning_rate: float = 5e-6
    warmup_steps: int = 700  # ~15% of max_steps
    per_device_train_batch_size: int = 32
    per_device_eval_batch_size: int = 16
    gradient_accumulation_steps: int = 2
    run_name: str = "whisper-dutch-production"


@dataclass
class LowMemoryConfig(TrainingConfig):
    """Configuration for systems with limited GPU memory"""

    max_steps: int = 7000
    eval_steps: int = 1000
    save_steps: int = 1000
    per_device_train_batch_size: int = 10
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    gradient_checkpointing: bool = True
    fp16: bool = True
    dataloader_num_workers: int = 8
    run_name: str = "whisper-dutch-low-memory"
