#!/usr/bin/env python3
"""
Shared dataset loading functionality for Whisper training and evaluation
"""

import json
import logging
import gc
from pathlib import Path
from typing import Optional, Callable

from datasets import Dataset, DatasetDict, Audio

logger = logging.getLogger(__name__)


class DatasetLoader:
    """Shared dataset loader for Whisper training and evaluation"""

    def __init__(
        self,
        dataset_path: str,
        dataset_file: str,
        min_duration_seconds: float = 0.5,
        max_duration_seconds: float = 30.0,
        dataset_seed: int = 1337,
        preprocessed_cache_dir: Optional[str] = None,
    ):
        self.dataset_path = Path(dataset_path)
        self.dataset_file = dataset_file
        self.min_duration_seconds = min_duration_seconds
        self.max_duration_seconds = max_duration_seconds
        self.dataset_seed = dataset_seed
        self.preprocessed_cache_dir = preprocessed_cache_dir

    def load_raw_dataset(self) -> Dataset:
        """Load the raw dataset from JSON file"""
        whisper_dataset_file = self.dataset_path / self.dataset_file

        if not whisper_dataset_file.exists():
            raise FileNotFoundError(f"Dataset file not found: {whisper_dataset_file}")

        logger.info(f"Loading dataset from {whisper_dataset_file}")

        # Load the JSON data
        with open(whisper_dataset_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Convert relative paths to absolute paths
        for item in data:
            audio_path = self.dataset_path / item["audio"]
            if not audio_path.exists():
                logger.warning(f"Audio file not found: {audio_path}")
            item["audio"] = str(audio_path)

        # Filter out items with missing audio files
        data = [item for item in data if Path(item["audio"]).exists()]
        logger.info(f"Loaded {len(data)} valid audio samples")

        # Create dataset
        dataset = Dataset.from_list(data)

        # Cast audio column
        dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

        return dataset

    def create_train_eval_split(
        self, dataset: Dataset, train_split_ratio: float = 0.9
    ) -> DatasetDict:
        """Create train/eval split from dataset"""
        train_size = int(train_split_ratio * len(dataset))
        eval_size = len(dataset) - train_size

        dataset_split = dataset.train_test_split(
            train_size=train_size, test_size=eval_size, seed=self.dataset_seed
        )

        # Create DatasetDict with proper naming
        dataset_dict = DatasetDict(
            {"train": dataset_split["train"], "validation": dataset_split["test"]}
        )

        logger.info(
            f"Dataset split (seed={self.dataset_seed}) - Train: {len(dataset_dict['train'])}, Validation: {len(dataset_dict['validation'])}"
        )

        return dataset_dict

    def load_dataset_for_training(
        self,
        max_train_samples: Optional[int] = None,
        max_eval_samples: Optional[int] = None,
        train_split_ratio: float = 0.9,
    ) -> DatasetDict:
        """Load and prepare dataset for training with optional sample limits"""
        dataset = self.load_raw_dataset()
        dataset_dict = self.create_train_eval_split(dataset, train_split_ratio)

        # Apply sample limits if specified
        if max_train_samples:
            dataset_dict["train"] = dataset_dict["train"].select(
                range(min(max_train_samples, len(dataset_dict["train"])))
            )
            logger.info(
                f"Limited train dataset to {len(dataset_dict['train'])} samples"
            )

        if max_eval_samples:
            dataset_dict["validation"] = dataset_dict["validation"].select(
                range(min(max_eval_samples, len(dataset_dict["validation"])))
            )
            logger.info(
                f"Limited validation dataset to {len(dataset_dict['validation'])} samples"
            )

        logger.info(
            f"Final dataset sizes - Train: {len(dataset_dict['train'])}, Validation: {len(dataset_dict['validation'])}"
        )

        return dataset_dict

    def load_preprocessed_dataset(
        self,
        max_train_samples: Optional[int] = None,
        max_eval_samples: Optional[int] = None,
    ) -> Optional[DatasetDict]:
        """Try to load preprocessed dataset from cache"""
        if not self.preprocessed_cache_dir:
            return None

        cache_path = Path(self.preprocessed_cache_dir)
        if not cache_path.exists():
            return None

        logger.info(f"Loading preprocessed dataset from cache: {cache_path}")
        try:
            cached_dataset = DatasetDict.load_from_disk(str(cache_path))

            # Apply sample limits to cached dataset
            if max_train_samples and len(cached_dataset["train"]) > max_train_samples:
                cached_dataset["train"] = cached_dataset["train"].select(
                    range(max_train_samples)
                )
                logger.info(f"Limited train dataset to {max_train_samples} samples")

            if (
                max_eval_samples
                and len(cached_dataset["validation"]) > max_eval_samples
            ):
                cached_dataset["validation"] = cached_dataset["validation"].select(
                    range(max_eval_samples)
                )
                logger.info(f"Limited validation dataset to {max_eval_samples} samples")

            logger.info(
                f"Using cached dataset - Train: {len(cached_dataset['train'])}, Validation: {len(cached_dataset['validation'])}"
            )
            return cached_dataset
        except Exception as e:
            logger.warning(
                f"Failed to load cache: {e}. Proceeding with fresh preprocessing..."
            )
            return None

    def preprocess_dataset(
        self,
        dataset_dict: DatasetDict,
        preprocessing_function: Callable,
        dataloader_num_workers: int = 4,
    ) -> DatasetDict:
        """Preprocess dataset and save to cache"""
        logger.info("Preprocessing dataset...")

        # Process datasets
        processed_dataset_dict = DatasetDict()

        for split_name, split_dataset in dataset_dict.items():
            logger.info(f"Processing {split_name} split...")
            processed_dataset = split_dataset.map(
                preprocessing_function,
                remove_columns=split_dataset.column_names,
                desc=f"Preprocessing {split_name}",
                # Use multiple processes for faster preprocessing
                num_proc=dataloader_num_workers if dataloader_num_workers > 1 else None,
                keep_in_memory=False,
                batch_size=1,
            )
            processed_dataset_dict[split_name] = processed_dataset

            # Force garbage collection after processing each split
            gc.collect()

            logger.info(f"Completed processing {split_name} split")

        # Save preprocessed dataset to cache if specified
        if self.preprocessed_cache_dir:
            cache_path = Path(self.preprocessed_cache_dir)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Saving preprocessed dataset to cache: {cache_path}")
            processed_dataset_dict.save_to_disk(str(cache_path))

        gc.collect()

        return processed_dataset_dict
