"""
Configuration file for Whisper Large-v3 fine-tuning
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingConfig:
    """Base training configuration parameters for Whisper Large-v3"""

    # Model settings - Fixed to whisper-large-v3
    model_name: str = "openai/whisper-large-v3"
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
    output_dir: str = "./whisper-large-v3-dutch-cgn"
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8  # Effective batch size of 16
    learning_rate: float = 5e-6  # Lower learning rate for large model
    warmup_steps: int = 500
    max_steps: int = 4500

    # Training settings optimized for whisper-large-v3
    gradient_checkpointing: bool = True
    bf16: bool = True
    dataloader_num_workers: int = 4

    # Evaluation and logging
    eval_strategy: str = "steps"
    eval_steps: int = 750
    save_steps: int = 750
    logging_steps: int = 25
    save_total_limit: int = 3

    # Generation settings optimized for whisper-large-v3
    generation_max_length: int = 448

    # Monitoring
    report_to: list = None
    push_to_hub: bool = False
    run_name: Optional[str] = None

    def __post_init__(self):
        if self.report_to is None:
            self.report_to = ["tensorboard"]


@dataclass
class QuickTestConfig(TrainingConfig):
    """Quick test configuration for whisper-large-v3 with low memory and small dataset"""

    output_dir: str = "../data/whisper-large-v3-dutch-cgn-test"
    max_steps: int = 300
    eval_steps: int = 50
    save_steps: int = 50
    max_train_samples: int = 200
    max_eval_samples: int = 30
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 2
    learning_rate: float = 3e-6
    warmup_steps: int = 50
    dataloader_num_workers: int = 8
    run_name: str = "whisper-large-v3-dutch-quick-test"


@dataclass
class ProductionConfig(TrainingConfig):
    """Production configuration for whisper-large-v3 with optimized parameters"""

    output_dir: str = "../data/whisper-large-v3-dutch-cgn-prod"
    max_steps: int = 4500  # ~3 epochs for 90k samples
    eval_steps: int = 750  # Evaluate ~6 times per epoch
    save_steps: int = 750  # Save ~6 times per epoch
    learning_rate: float = 5e-6
    warmup_steps: int = 700  # ~15% of max_steps
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 2
    dataloader_num_workers: int = 8
    run_name: str = "whisper-large-v3-dutch-prod"


# Default configuration
DEFAULT_CONFIG = QuickTestConfig()
