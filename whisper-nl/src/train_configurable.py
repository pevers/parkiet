#!/usr/bin/env python3
"""
Configurable Whisper Fine-tuning Script for Dutch/Flemish CGN Data
"""

import logging
import argparse
from typing import Dict, List, Any
import torch
from datasets import DatasetDict
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
import evaluate

from config import TrainingConfig, QuickTestConfig, ProductionConfig, LowMemoryConfig
from text_normalizer import BasicTextNormalizer
from dataset_loader import DatasetLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class WhisperDataCollator:
    """Data collator for Whisper training"""

    def __init__(self, processor, decoder_start_token_id):
        self.processor = processor
        self.decoder_start_token_id = decoder_start_token_id

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Split inputs and labels
        model_input_name = self.processor.model_input_names[0]
        input_features = [feature[model_input_name] for feature in features]
        label_features = [feature["labels"] for feature in features]

        batch = self.processor.feature_extractor.pad(
            [{"input_features": feature} for feature in input_features],
            return_tensors="pt",
        )

        # Pad label features with explicit padding token
        labels_batch = self.processor.tokenizer.pad(
            [{"input_ids": feature} for feature in label_features], return_tensors="pt"
        )

        # Replace padding with -100 to ignore in loss calculation
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # Cut decoder_start_token_id if present
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


class ConfigurableWhisperTrainer:
    """Configurable trainer class for Whisper fine-tuning on CGN data"""

    def __init__(self, config: TrainingConfig):
        self.config = config

        # Initialize processor and model
        self.processor = None
        self.model = None
        self.feature_extractor = None
        self.tokenizer = None

        # Text normalizer for evaluation
        self.normalizer = BasicTextNormalizer()

        # Metrics
        self.wer_metric = evaluate.load("wer")

        self.setup_model_and_processor()

    def setup_model_and_processor(self):
        """Initialize model, tokenizer, and feature extractor"""
        logger.info(f"Loading model and processor: {self.config.model_name}")

        # Load feature extractor
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(
            self.config.model_name
        )

        # Load tokenizer
        self.tokenizer = WhisperTokenizer.from_pretrained(
            self.config.model_name,
            language=self.config.language,
            task=self.config.task,
        )

        # Load processor
        self.processor = WhisperProcessor.from_pretrained(
            self.config.model_name,
            language=self.config.language,
            task=self.config.task,
        )

        # Ensure forced_decoder_ids are properly cleared in processor
        self.processor.tokenizer.set_prefix_tokens(
            language=self.config.language, task=self.config.task
        )

        # Load model
        self.model = WhisperForConditionalGeneration.from_pretrained(
            self.config.model_name
        )

        # Resize token embeddings if needed
        if len(self.tokenizer) > self.model.config.vocab_size:
            self.model.resize_token_embeddings(len(self.tokenizer))

        # Model configuration
        self.model.generation_config.forced_decoder_ids = None
        self.model.generation_config.suppress_tokens = []
        self.model.generation_config.language = self.config.language
        self.model.generation_config.task = self.config.task

        # Disable caching when gradient checkpointing is enabled
        if self.config.gradient_checkpointing:
            self.model.config.use_cache = False
            self.model.generation_config.use_cache = False

        # Freeze parts of model if specified
        if self.config.freeze_feature_encoder:
            self.model.freeze_feature_encoder()
            logger.info("Feature encoder frozen")

        if self.config.freeze_encoder:
            self.model.freeze_encoder()
            logger.info("Encoder frozen")

        logger.info("Model and processor loaded successfully")

    def load_dataset(self) -> DatasetDict:
        """Load and prepare the CGN dataset using dataset loader"""
        self.dataset_loader = DatasetLoader(
            dataset_path=self.config.dataset_path,
            dataset_file=self.config.dataset_file,
            min_duration_seconds=self.config.min_duration_seconds,
            max_duration_seconds=self.config.max_duration_seconds,
            dataset_seed=self.config.dataset_seed,
            preprocessed_cache_dir=self.config.preprocessed_cache_dir,
        )

        return self.dataset_loader.load_dataset_for_training(
            max_train_samples=self.config.max_train_samples,
            max_eval_samples=self.config.max_eval_samples,
        )

    def preprocess_dataset(self, dataset_dict: DatasetDict) -> DatasetDict:
        """Preprocess the dataset for training"""
        # Check if we can load from cache first
        cached_dataset = self.dataset_loader.load_preprocessed_dataset(
            max_train_samples=self.config.max_train_samples,
            max_eval_samples=self.config.max_eval_samples,
        )

        if cached_dataset is not None:
            return cached_dataset

        def prepare_dataset(batch):
            # Load and process audio
            audio = batch["audio"]

            # Compute input features
            batch["input_features"] = self.feature_extractor(
                [x["array"] for x in audio],
                sampling_rate=audio[0]["sampling_rate"],
            ).input_features

            # Tokenize text
            batch["labels"] = self.tokenizer(batch["text"]).input_ids

            # NOTE: This squeezes the speed a LOT but necessary on my tiny desktop PC
            # del audio
            # gc.collect()

            return batch

        # Preprocess and cache dataset
        return self.dataset_loader.preprocess_dataset(
            dataset_dict, prepare_dataset, self.config.dataloader_num_workers
        )

    def compute_metrics(self, eval_preds):
        """Compute WER metric"""
        pred_ids, label_ids = eval_preds

        # Replace -100 with pad token id for proper decoding
        label_ids[label_ids == -100] = self.tokenizer.pad_token_id

        # Decode predictions and labels with proper attention handling
        pred_str = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = self.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        # Normalize texts
        pred_str = [self.normalizer(pred) for pred in pred_str]
        label_str = [self.normalizer(label) for label in label_str]

        # Compute WER
        wer = 100 * self.wer_metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}

    def create_trainer(self, train_dataset, eval_dataset):
        """Create and configure the trainer"""

        # Training arguments from config
        training_args = Seq2SeqTrainingArguments(
            output_dir=self.config.output_dir,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            max_steps=self.config.max_steps,
            gradient_checkpointing=self.config.gradient_checkpointing,
            fp16=self.config.fp16,
            eval_strategy=self.config.eval_strategy,
            eval_steps=self.config.eval_steps,
            save_steps=self.config.save_steps,
            logging_steps=self.config.logging_steps,
            report_to=self.config.report_to,
            run_name=self.config.run_name,
            load_best_model_at_end=True,
            metric_for_best_model="wer",
            greater_is_better=False,
            push_to_hub=self.config.push_to_hub,
            dataloader_num_workers=self.config.dataloader_num_workers,
            save_total_limit=self.config.save_total_limit,
            predict_with_generate=True,
            generation_max_length=self.config.generation_max_length,
        )

        # Data collator
        data_collator = WhisperDataCollator(
            processor=self.processor,
            decoder_start_token_id=self.model.config.decoder_start_token_id,
        )

        # Create trainer
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            tokenizer=self.processor.feature_extractor,
        )

        return trainer

    def train(self):
        """Main training function"""
        logger.info("Starting Whisper fine-tuning...")
        logger.info(f"Configuration: {self.config}")

        # Load and preprocess dataset
        raw_dataset = self.load_dataset()
        processed_dataset = self.preprocess_dataset(raw_dataset)

        # Create trainer
        trainer = self.create_trainer(
            train_dataset=processed_dataset["train"],
            eval_dataset=processed_dataset["validation"],
        )

        # Train
        logger.info("Starting training...")
        trainer.train()

        # Save final model
        logger.info(f"Saving final model to {self.config.output_dir}")
        trainer.save_model()
        self.processor.save_pretrained(self.config.output_dir)

        # Final evaluation
        logger.info("Running final evaluation...")
        eval_results = trainer.evaluate()

        logger.info("Training completed!")
        logger.info(f"Final WER: {eval_results['eval_wer']:.2f}%")

        return trainer, eval_results


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description="Fine-tune Whisper on CGN data")
    parser.add_argument(
        "--config",
        type=str,
        choices=["quick", "production", "low_memory"],
        default="quick",
        help="Configuration preset to use",
    )
    parser.add_argument("--output-dir", type=str, help="Override output directory")
    parser.add_argument(
        "--max-train-samples", type=int, help="Limit number of training samples"
    )
    parser.add_argument(
        "--max-eval-samples", type=int, help="Limit number of evaluation samples"
    )
    parser.add_argument("--run-name", type=str, help="Custom name for TensorBoard run")

    args = parser.parse_args()

    # Select configuration
    if args.config == "quick":
        config = QuickTestConfig()
        logger.info("Using quick test configuration")
    elif args.config == "production":
        config = ProductionConfig()
        logger.info("Using production configuration")
    elif args.config == "low_memory":
        config = LowMemoryConfig()
        logger.info("Using low memory configuration")

    # Apply command line overrides
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.max_train_samples:
        config.max_train_samples = args.max_train_samples
    if args.max_eval_samples:
        config.max_eval_samples = args.max_eval_samples
    if args.run_name:
        config.run_name = args.run_name

    if not torch.cuda.is_available():
        logger.warning("CUDA not available. Training is not possible on CPU.")
        return

    logger.info(f"CUDA available. Using GPU: {torch.cuda.get_device_name()}")
    logger.info(
        f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
    )

    # Create trainer and start training
    trainer = ConfigurableWhisperTrainer(config)
    trainer.train()

    logger.info("Training pipeline completed successfully!")


if __name__ == "__main__":
    main()
