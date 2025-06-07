#!/usr/bin/env python3
"""
Evaluation script to compare two Whisper checkpoints
"""

import logging
import argparse
from pathlib import Path
import torch
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
)
import evaluate

from text_normalizer import BasicTextNormalizer
from dataset_loader import DatasetLoader
from config import TrainingConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class CheckpointEvaluator:
    """Evaluator for comparing two Whisper checkpoints"""

    def __init__(
        self,
        checkpoint1_path: str,
        checkpoint2_path: str,
        config: TrainingConfig = None,
    ):
        self.checkpoint1_path = checkpoint1_path
        self.checkpoint2_path = checkpoint2_path
        self.config = config or TrainingConfig()  # Use default config if none provided

        # Text normalizer for evaluation
        self.normalizer = BasicTextNormalizer()

        # WER metric
        self.wer_metric = evaluate.load("wer")

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Load models and processors
        self.processor1, self.model1 = self.load_checkpoint(
            checkpoint1_path, "Checkpoint 1"
        )
        self.processor2, self.model2 = self.load_checkpoint(
            checkpoint2_path, "Checkpoint 2"
        )

    def load_checkpoint(self, checkpoint_path: str, name: str):
        """Load a checkpoint and its processor"""
        logger.info(f"Loading {name} from {checkpoint_path}...")

        # Load model
        model = WhisperForConditionalGeneration.from_pretrained(checkpoint_path)
        model.to(self.device)
        model.eval()

        # Load processor - use the same settings as train_configurable.py
        processor = WhisperProcessor.from_pretrained(
            "openai/whisper-small", language="dutch", task="transcribe"
        )

        # Configure generation
        model.generation_config.suppress_tokens = []
        model.generation_config.language = "dutch"
        model.generation_config.task = "transcribe"

        logger.info(f"{name} loaded successfully")
        return processor, model

    def load_evaluation_dataset(
        self, dataset_path: str, dataset_file: str, num_samples: int = 10
    ):
        """Load evaluation dataset and sample random examples"""
        dataset_loader = DatasetLoader(
            dataset_path=dataset_path,
            dataset_file=dataset_file,
            min_duration_seconds=self.config.min_duration_seconds,
            max_duration_seconds=self.config.max_duration_seconds,
            dataset_seed=self.config.dataset_seed,
            preprocessed_cache_dir=self.config.preprocessed_cache_dir,
        )

        return dataset_loader.load_preprocessed_dataset(
            num_samples=num_samples,
            evaluation_seed=self.config.dataset_seed,
        )

    def transcribe_with_model(self, audio_data, processor, model):
        """Transcribe audio with a given model"""
        # Process audio
        inputs = processor(
            audio_data["array"],
            sampling_rate=audio_data["sampling_rate"],
            return_tensors="pt",
        )
        input_features = inputs.input_features.to(self.device)

        # Generate transcription
        with torch.no_grad():
            predicted_ids = model.generate(input_features, max_length=225)

        # Decode transcription
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
        return transcription[0]

    def evaluate_samples(self, dataset):
        """Evaluate both models on the dataset samples"""
        results = []

        for i, sample in enumerate(dataset):
            logger.info(f"Processing sample {i + 1}/{len(dataset)}")

            # Get reference text
            reference = sample["text"]

            # Transcribe with both models
            transcription1 = self.transcribe_with_model(
                sample["audio"], self.processor1, self.model1
            )
            transcription2 = self.transcribe_with_model(
                sample["audio"], self.processor2, self.model2
            )

            # Normalize all texts
            norm_reference = self.normalizer(reference)
            norm_transcription1 = self.normalizer(transcription1)
            norm_transcription2 = self.normalizer(transcription2)

            # Compute WER for each model
            wer1 = self.wer_metric.compute(
                predictions=[norm_transcription1], references=[norm_reference]
            )
            wer2 = self.wer_metric.compute(
                predictions=[norm_transcription2], references=[norm_reference]
            )

            result = {
                "sample_id": i,
                "reference": reference,
                "transcription1": transcription1,
                "transcription2": transcription2,
                "norm_reference": norm_reference,
                "norm_transcription1": norm_transcription1,
                "norm_transcription2": norm_transcription2,
                "wer1": wer1 * 100,
                "wer2": wer2 * 100,
            }

            results.append(result)

            # Print sample results
            print(f"\n--- Sample {i + 1} ---")
            print(f"Reference: '{reference}'")
            print(f"Checkpoint 1: '{transcription1}' (WER: {wer1 * 100:.2f}%)")
            print(f"Checkpoint 2: '{transcription2}' (WER: {wer2 * 100:.2f}%)")
            print(f"Normalized Reference: '{norm_reference}'")
            print(f"Normalized Checkpoint 1: '{norm_transcription1}'")
            print(f"Normalized Checkpoint 2: '{norm_transcription2}'")

        return results

    def print_summary(self, results):
        """Print evaluation summary"""
        wer1_scores = [r["wer1"] for r in results]
        wer2_scores = [r["wer2"] for r in results]

        avg_wer1 = sum(wer1_scores) / len(wer1_scores)
        avg_wer2 = sum(wer2_scores) / len(wer2_scores)

        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Checkpoint 1 ({self.checkpoint1_path}):")
        print(f"  Average WER: {avg_wer1:.2f}%")
        print(f"  Individual WERs: {[f'{w:.2f}%' for w in wer1_scores]}")

        print(f"\nCheckpoint 2 ({self.checkpoint2_path}):")
        print(f"  Average WER: {avg_wer2:.2f}%")
        print(f"  Individual WERs: {[f'{w:.2f}%' for w in wer2_scores]}")

        print(f"\nComparison:")
        if avg_wer1 < avg_wer2:
            print(f"  Checkpoint 1 is BETTER by {avg_wer2 - avg_wer1:.2f}% WER")
        elif avg_wer2 < avg_wer1:
            print(f"  Checkpoint 2 is BETTER by {avg_wer1 - avg_wer2:.2f}% WER")
        else:
            print(f"  Both checkpoints have the same average WER")

        # Count wins
        wins1 = sum(1 for r in results if r["wer1"] < r["wer2"])
        wins2 = sum(1 for r in results if r["wer2"] < r["wer1"])
        ties = sum(1 for r in results if r["wer1"] == r["wer2"])

        print(f"\nSample-by-sample comparison:")
        print(f"  Checkpoint 1 wins: {wins1}/{len(results)} samples")
        print(f"  Checkpoint 2 wins: {wins2}/{len(results)} samples")
        print(f"  Ties: {ties}/{len(results)} samples")


def main():
    parser = argparse.ArgumentParser(description="Compare two Whisper checkpoints")
    parser.add_argument("--checkpoint1", required=True, help="Path to first checkpoint")
    parser.add_argument(
        "--checkpoint2", required=True, help="Path to second checkpoint"
    )
    parser.add_argument(
        "--dataset-path", default="../data/training", help="Path to dataset directory"
    )
    parser.add_argument(
        "--dataset-file", default="whisper_dataset.json", help="Dataset JSON file name"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of random samples to evaluate",
    )

    args = parser.parse_args()

    # Verify checkpoint paths exist
    if not Path(args.checkpoint1).exists():
        raise FileNotFoundError(f"Checkpoint 1 not found: {args.checkpoint1}")
    if not Path(args.checkpoint2).exists():
        raise FileNotFoundError(f"Checkpoint 2 not found: {args.checkpoint2}")

    # Create config for consistent dataset handling
    config = TrainingConfig()

    # Create evaluator
    evaluator = CheckpointEvaluator(args.checkpoint1, args.checkpoint2, config)

    # Load evaluation samples
    eval_dataset = evaluator.load_evaluation_dataset(
        args.dataset_path, args.dataset_file, args.num_samples
    )

    # Run evaluation
    results = evaluator.evaluate_samples(eval_dataset)

    # Print summary
    evaluator.print_summary(results)


if __name__ == "__main__":
    main()
