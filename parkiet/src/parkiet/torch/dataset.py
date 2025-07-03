import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset
from pathlib import Path
import logging

from parkiet.dia.config import DiaConfig
from parkiet.dia.model import Dia

log = logging.getLogger(__name__)


class TestSampleDataset(Dataset):
    """Inefficient data sampler just to check the training loop."""

    def __init__(self, samples: list[tuple[torch.Tensor, torch.Tensor]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        audio_input, encoded_prompt = self.samples[idx]
        return {
            "text": encoded_prompt,
            "audio": audio_input,
        }


class AudioTextDataset(Dataset):
    """PyTorch dataset for loading audio-text pairs from parquet files created by the chunker."""

    def __init__(
        self,
        parquet_path: str | Path,
        config: DiaConfig,
        max_audio_length: int | None = None,
        max_text_length: int | None = None,
    ):
        """
        Initialize the dataset.

        Args:
            parquet_path: Path to the parquet file containing processed chunks
            config: DiaConfig object containing model configuration
            max_audio_length: Maximum audio sequence length (in tokens). If None, uses config value.
        """
        self.parquet_path = Path(parquet_path)
        self.config = config
        self.max_audio_length = max_audio_length or config.data.audio_length
        self.max_text_length = max_text_length or config.data.text_length
        if not self.parquet_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {self.parquet_path}")

        # Load the dataframe
        log.info(f"Loading dataset from {self.parquet_path}")
        self.df = pd.read_parquet(self.parquet_path)

        # Filter out rows with invalid data
        self.df = self.df[
            (self.df["transcription"].notna())
            & (self.df["transcription"].str.len() > 0)
            & (self.df["encoded_audio"].notna())
        ].reset_index(drop=True)

        log.info(f"Loaded {len(self.df)} valid samples from dataset")

        # Get data configuration values
        self.text_length = config.data.text_length
        self.text_pad_value = config.data.text_pad_value
        self.audio_pad_value = config.data.audio_pad_value
        self.channels = config.data.channels

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Get a single sample from the dataset.

        Args:
            idx: Index of the sample

        Returns:
            Dictionary containing 'text' and 'audio' tensors
        """
        row = self.df.iloc[idx]

        # Process text
        text_tokens = self._encode_text(row["transcription"])

        # Process audio
        audio_tokens = self._decode_audio(
            row["encoded_audio"], row["encoded_audio_shape"]
        )

        return {
            "text": text_tokens,
            "audio": audio_tokens,
        }

    def _encode_text(self, text: str) -> torch.Tensor:
        """
        Encode text using byte-level encoding (same as DIA model).

        Args:
            text: Input text string

        Returns:
            Encoded text tensor of shape [text_length]
        """
        # Convert to bytes and replace special tokens
        byte_text = text.encode("utf-8")
        # TODO: Data contains speaker tags up to [S4], we should fix this!
        replaced_bytes = (
            byte_text.replace(b"[S1]", b"\x01")
            .replace(b"[S2]", b"\x02")
            .replace(b"[S3]", b"\x03")
            .replace(b"[S4]", b"\x04")
        )
        text_tokens = list(replaced_bytes)
        return torch.tensor(text_tokens[: self.max_text_length], dtype=torch.long)

    def _decode_audio(
        self, encoded_audio: np.ndarray, encoded_audio_shape: np.ndarray
    ) -> torch.Tensor:
        """
        Decode flattened audio data back to its original tensor shape.

        Args:
            encoded_audio: Flattened audio data as list of integers
            encoded_audio_shape: Original shape of the audio tensor [T, C]

        Returns:
            Audio tensor of shape [max_audio_length, channels]
        """
        # Reconstruct the original tensor
        audio_tensor = torch.tensor(encoded_audio, dtype=torch.long).reshape(
            encoded_audio_shape.tolist()
        )

        # Ensure it has the expected number of channels
        if audio_tensor.shape[-1] != self.channels:
            raise ValueError(
                f"Audio tensor has {audio_tensor.shape[-1]} channels, "
                f"but config expects {self.channels} channels"
            )
        return audio_tensor

    def get_sample_info(self, idx: int) -> dict[str, any]:
        """
        Get metadata information for a sample.

        Args:
            idx: Index of the sample

        Returns:
            Dictionary with sample metadata
        """
        row = self.df.iloc[idx]
        return {
            "source_file": row["source_file"],
            "chunk_id": row["chunk_id"],
            "start_ms": row["start_ms"],
            "end_ms": row["end_ms"],
            "duration_ms": row["duration_ms"],
            "transcription": row["transcription"],
            "original_audio_shape": row["encoded_audio_shape"],
        }


def create_dataloader(
    parquet_path: str | Path,
    config: DiaConfig,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    max_audio_length: int | None = None,
    max_text_length: int | None = None,
) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader for the AudioTextDataset.

    Args:
        parquet_path: Path to the parquet file
        config: DiaConfig object
        batch_size: Batch size for training
        shuffle: Whether to shuffle the dataset
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        max_audio_length: Maximum audio sequence length

    Returns:
        DataLoader instance
    """
    dataset = AudioTextDataset(
        parquet_path=parquet_path,
        config=config,
        max_audio_length=max_audio_length,
        max_text_length=max_text_length,
    )

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,  # Drop last incomplete batch for consistent training
    )


def create_test_dataloader(
    dia: Dia,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 1,
    pin_memory: bool = True,
) -> torch.utils.data.DataLoader:
    samples = []
    for file in Path("samples").glob("*.txt"):
        with open(file, "r") as f:
            prompt = f.read()
        encoded_audio = dia.load_audio(file.with_suffix(".mp3")).cpu()
        encoded_prompt = dia._encode_text(prompt).cpu()
        samples.append((encoded_audio, encoded_prompt))
    dataset = TestSampleDataset(samples)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
