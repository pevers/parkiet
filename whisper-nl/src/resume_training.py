"""
Resume Whisper training from the last checkpoint
"""

import argparse
import logging
from pathlib import Path
from train_configurable import (
    ConfigurableWhisperTrainer,
    ProductionConfig,
    QuickTestConfig,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def find_latest_checkpoint(output_dir):
    """Find the latest checkpoint in the output directory"""
    output_path = Path(output_dir)
    if not output_path.exists():
        return None

    checkpoints = []
    for item in output_path.iterdir():
        if item.is_dir() and item.name.startswith("checkpoint-"):
            try:
                step_num = int(item.name.split("-")[1])
                checkpoints.append((step_num, item))
            except:
                logger.warning(
                    f"Skipping checkpoint {item.name} because it is not a valid checkpoint"
                )
                continue

    if not checkpoints:
        return None

    # Return the checkpoint with the highest step number
    latest_step, latest_checkpoint = max(checkpoints, key=lambda x: x[0])
    return str(latest_checkpoint)


def main():
    parser = argparse.ArgumentParser(
        description="Resume Whisper training from checkpoint"
    )
    parser.add_argument(
        "--config",
        type=str,
        choices=["quick", "production"],
        default="production",
        help="Configuration preset to use",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        help="Specific checkpoint directory to resume from",
    )

    args = parser.parse_args()

    # Select configuration
    if args.config == "quick":
        config = QuickTestConfig()
    else:
        config = ProductionConfig()

    # Find checkpoint to resume from
    if args.checkpoint_dir:
        resume_from_checkpoint = args.checkpoint_dir
    else:
        resume_from_checkpoint = find_latest_checkpoint(config.output_dir)

    if resume_from_checkpoint:
        logger.info(f"Resuming training from checkpoint: {resume_from_checkpoint}")
    else:
        logger.info("No checkpoint found, starting fresh training")

    # Create trainer
    trainer_instance = ConfigurableWhisperTrainer(config)

    # Load and preprocess dataset
    raw_dataset = trainer_instance.load_dataset()
    processed_dataset = trainer_instance.preprocess_dataset(raw_dataset)

    # Create trainer
    trainer = trainer_instance.create_trainer(
        train_dataset=processed_dataset["train"],
        eval_dataset=processed_dataset["validation"],
    )

    # Resume training
    logger.info("Starting/resuming training...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Save final model
    logger.info(f"Saving final model to {config.output_dir}")
    trainer.save_model()
    trainer_instance.processor.save_pretrained(config.output_dir)

    # Final evaluation
    logger.info("Running final evaluation...")
    eval_results = trainer.evaluate()

    logger.info("Training completed!")
    logger.info(f"Final WER: {eval_results.get('eval_wer', 'N/A')}")


if __name__ == "__main__":
    main()
