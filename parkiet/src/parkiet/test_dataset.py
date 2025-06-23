import math
from parkiet.dataset import AudioTextDataset
from parkiet.dia.model import Dia, ComputeDtype
from parkiet.train import TrainingConfig


def main():
    """
    Calculates and prints dataset size and steps per epoch.
    """
    print("Loading configuration and dataset...")

    training_config = TrainingConfig()

    # We only need the config, so we don't need to load DAC or use GPU
    dia = Dia.from_local(
        config_path="config.json",
        # We don't need weights for this, but from_local requires it.
        # An empty path might fail, so using the one from train.py
        checkpoint_path="weights/dia-v0_1.pth",
        compute_dtype=ComputeDtype.FLOAT32,  # float32 is fine for this
        device="cpu",
        load_dac=False,
    )

    dataset = AudioTextDataset(
        parquet_path="./data/chunks_dataset.parquet",
        config=dia.config,
    )

    dataset_size = len(dataset)
    batch_size = training_config.batch_size
    gradient_accumulation_steps = training_config.gradient_accumulation_steps

    # The dataloader uses drop_last=True, so we use math.floor
    steps_per_epoch = math.floor(dataset_size / batch_size)
    optimizer_steps_per_epoch = math.floor(
        steps_per_epoch / gradient_accumulation_steps
    )

    print("\n--- Dataset Information ---")
    print(f"Total dataset size: {dataset_size} samples")
    print(f"Batch size: {batch_size}")
    print(f"Gradient Accumulation Steps: {gradient_accumulation_steps}")
    print(f"Steps per epoch (dataloader iterations): {steps_per_epoch}")
    print(f"Optimizer steps per epoch: {optimizer_steps_per_epoch}")
    print("---------------------------\n")


if __name__ == "__main__":
    main()
