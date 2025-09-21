import numpy as np
import pandas as pd
from pathlib import Path
import logging
from collections.abc import Iterator
from parkiet.dia.config import DiaConfig

log = logging.getLogger(__name__)


class AudioTextDataset:
    """Simple dataset for loading audio-text pairs from parquet files."""

    def __init__(
        self,
        parquet_path: str | Path,
        config: DiaConfig,
        max_audio_length: int | None = None,
        max_text_length: int | None = None,
        transcription_clean_prob: float = 0.1,
        text_dropout_prob: float = 0.15,
    ):
        """
        Initialize the dataset.

        Args:
            parquet_path: Folder path to local parquet files
            config: DiaConfig object containing model configuration
            max_audio_length: Maximum audio sequence length (in tokens). If None, uses config value.
            max_text_length: Maximum text sequence length (in tokens). If None, uses config value.
            transcription_clean_prob: Probability of using transcription_clean instead of transcription
            text_dropout_prob: Probability of dropping out text for CFG
        """
        self.config = config
        self.max_audio_length = (
            max_audio_length or config.decoder_config.max_position_embeddings
        )
        self.max_text_length = (
            max_text_length or config.encoder_config.max_position_embeddings
        )
        self.transcription_clean_prob = transcription_clean_prob
        self.text_dropout_prob = text_dropout_prob
        self.rng = np.random.RandomState(42)

        # Load all parquet files
        self.parquet_files = discover_parquet_shards(parquet_path)
        log.info(
            f"Found {len(self.parquet_files)} parquet shards: {[Path(f).name for f in self.parquet_files]}"
        )

        self.current_shard_idx = 0
        self.df = None

        if self.parquet_files:  # Only load if we have shards assigned
            self.load_shard()

    def load_shard(self):
        """Load the current parquet shard."""
        if self.current_shard_idx < len(self.parquet_files):
            parquet_file = self.parquet_files[self.current_shard_idx]
            self.df = pd.read_parquet(parquet_file)
            log.info(
                f"Loaded shard {self.current_shard_idx}: {parquet_file} with {len(self.df)} samples"
            )
        else:
            self.df = pd.DataFrame()  # Empty dataframe when no more shards

    def reset(self):
        """Reset the dataset to the beginning for a new epoch."""
        self.current_shard_idx = 0
        if self.parquet_files:
            self.load_shard()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        """
        Get a single sample from the dataset.

        Args:
            idx: Index of the sample

        Returns:
            Dictionary containing 'text' and 'audio' NumPy arrays
        """
        row = self.df.iloc[idx]

        # Sample transcription field with probability
        if (
            self.rng.random() < self.transcription_clean_prob
            and "transcription_clean" in row
        ):
            transcript_text = row["transcription_clean"]
        else:
            transcript_text = row["transcription"]

        # Text dropout for CFG
        if self.rng.random() < self.text_dropout_prob:
            text_tokens = self._create_empty_text_sequence()
        else:
            text_tokens = self._encode_text(transcript_text)

        # Process audio
        audio_tokens = self._process_audio(
            row["encoded_audio"], row["encoded_audio_shape"]
        )

        # Get class-balanced weight if available, default to 1.0
        cb_weight = row.get("cb_weight", 1.0)

        return {
            "text": text_tokens,
            "audio": audio_tokens,
            "cb_weight": np.float32(cb_weight),
        }

    def _encode_text(self, text: str) -> np.ndarray:
        """
        Encode text using byte-level encoding and pad to max_text_length.

        Args:
            text: Input text string

        Returns:
            Encoded text array of shape [max_text_length] (padded)
        """
        # Convert to bytes and replace special tokens
        byte_text = text.encode("utf-8")
        replaced_bytes = (
            byte_text.replace(b"[S1]", b"\x01")
            .replace(b"[S2]", b"\x02")
            .replace(b"[S3]", b"\x03")
            .replace(b"[S4]", b"\x04")
        )
        text_tokens = list(replaced_bytes)

        # Truncate to max_text_length
        if len(text_tokens) > self.max_text_length:
            text_tokens = text_tokens[: self.max_text_length]

        # Pad to max_text_length
        if len(text_tokens) < self.max_text_length:
            pad_value = 0
            text_tokens.extend([pad_value] * (self.max_text_length - len(text_tokens)))

        return np.array(text_tokens, dtype=np.int32)

    def _decode_audio(self, encoded_audio, encoded_audio_shape) -> np.ndarray:
        """
        Decode flattened audio data back to its original tensor shape.

        Args:
            encoded_audio: Flattened audio data as list of integers
            encoded_audio_shape: Original shape of the audio tensor [T, C]

        Returns:
            Audio array of original shape [T, C]
        """
        return np.array(encoded_audio, dtype=np.int32).reshape(
            encoded_audio_shape.tolist()
        )

    def _process_audio(self, encoded_audio, encoded_audio_shape) -> np.ndarray:
        """
        Process audio: decode, add BOS/EOS tokens, and pad to max_audio_length.

        Args:
            encoded_audio: Flattened audio data as list of integers
            encoded_audio_shape: Original shape of the audio tensor [T, C]

        Returns:
            Audio sequence of shape [max_audio_length, C] with BOS,audio_tokens,EOS,PAD
        """
        # Decode raw audio tokens
        raw_audio_tokens = self._decode_audio(encoded_audio, encoded_audio_shape)

        # Truncate if too long (leave room for BOS and EOS)
        max_content_length = self.max_audio_length - 2  # Reserve space for BOS/EOS
        if raw_audio_tokens.shape[0] > max_content_length:
            raw_audio_tokens = raw_audio_tokens[:max_content_length]

        # Create sequence padded to max_audio_length
        audio_sequence = np.full(
            (self.max_audio_length, self.config.decoder_config.num_channels),
            self.config.pad_token_id,
            dtype=np.int32,
        )

        # Set BOS token at the beginning
        audio_sequence[0, :] = self.config.bos_token_id

        # Copy audio tokens
        content_length = raw_audio_tokens.shape[0]
        audio_sequence[1 : 1 + content_length, :] = raw_audio_tokens

        # Set EOS token after the content
        if 1 + content_length < self.max_audio_length:
            audio_sequence[1 + content_length, :] = self.config.eos_token_id

        return audio_sequence

    def _create_empty_text_sequence(self) -> np.ndarray:
        """
        Create an empty text sequence for CFG (classifier-free guidance).

        Returns:
            Text sequence of shape [max_text_length] filled with pad tokens
        """
        return np.zeros(self.max_text_length, dtype=np.int32)

    def get_raw_audio(self, idx: int) -> np.ndarray:
        """
        Get raw audio tokens without BOS/EOS/padding processing.
        This is useful for audio playback and analysis.

        Args:
            idx: Index of the sample

        Returns:
            Raw audio array of original shape [T, C]
        """
        row = self.df.iloc[idx]
        return self._decode_audio(row["encoded_audio"], row["encoded_audio_shape"])

    def batch_iterator(
        self,
        batch_size: int,
        shuffle: bool = True,
        seed: int = 42,
        use_sample_prob: bool = True,
    ) -> Iterator[dict[str, list]]:
        """
        Create a batch iterator that handles shard transitions.

        Args:
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle indices within each shard
            seed: Random seed for deterministic shuffling across processes
            use_sample_prob: Whether to use sample_prob for weighted sampling

        Yields:
            Batches of data as dictionaries with 'text', 'audio', and 'cb_weight' lists
        """
        rng = np.random.RandomState(seed)
        while self.current_shard_idx < len(self.parquet_files):
            # Get current shard size
            num_samples = len(self.df)

            if num_samples == 0:
                # Move to next shard if current is empty
                self.current_shard_idx += 1
                if self.current_shard_idx < len(self.parquet_files):
                    self.load_shard()
                continue

            # Create indices for current shard
            if use_sample_prob in self.df.columns:
                # Use weighted sampling based on sample_prob
                sample_probs = self.df["sample_prob"].values
                sample_probs = sample_probs / np.sum(sample_probs)  # Normalize
                indices = rng.choice(
                    num_samples, size=num_samples, replace=True, p=sample_probs
                )
                if shuffle:
                    indices = rng.permutation(indices)
            else:
                # Standard uniform sampling
                indices = np.arange(num_samples)
                if shuffle:
                    indices = rng.permutation(indices)

            # Process all samples in current shard
            for i in range(0, num_samples, batch_size):
                batch_indices = indices[i : i + batch_size]

                # Skip empty batches
                if len(batch_indices) == 0:
                    break

                # Skip partial batches to ensure consistent sharding
                if len(batch_indices) < batch_size:
                    break

                # Create batch (full size only)
                batch_data = {"text": [], "audio": [], "cb_weight": []}
                for idx in batch_indices:
                    sample = self[int(idx)]
                    batch_data["text"].append(sample["text"])
                    batch_data["audio"].append(sample["audio"])
                    batch_data["cb_weight"].append(sample["cb_weight"])

                yield batch_data

            # Move to next shard after processing current one
            self.current_shard_idx += 1
            if self.current_shard_idx < len(self.parquet_files):
                self.load_shard()


def discover_parquet_shards(shards_dir: str = "shards") -> list[str]:
    """
    Discover all parquet shard files in the shards directory.

    Args:
        shards_dir: Directory containing parquet shard files

    Returns:
        Sorted list of parquet file paths
    """
    shards_path = Path(shards_dir)
    if not shards_path.exists():
        raise FileNotFoundError(f"Shards directory not found: {shards_path}")

    parquet_files = list(shards_path.glob("*.parquet"))
    parquet_files.sort()  # Ensure consistent ordering

    return [str(f) for f in parquet_files]


def get_shard_size(parquet_file: str) -> int:
    """
    Get the number of samples in a single parquet shard.

    Args:
        parquet_file: Path to a single parquet file

    Returns:
        Number of samples in the shard
    """
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_file, columns=[])
    return table.num_rows


def get_total_samples_from_shards(parquet_files: list[str]) -> int:
    """
    Get the total number of samples across all parquet shards.

    Args:
        parquet_files: List of parquet file paths

    Returns:
        Total number of samples across all shards
    """
    total_samples = 0
    for parquet_file in parquet_files:
        # Read just the metadata to get row count efficiently
        # Use pyarrow instead of pandas to avoid empty columns bug
        import pyarrow.parquet as pq

        table = pq.read_table(parquet_file, columns=[])
        total_samples += table.num_rows
    return total_samples


def calculate_total_steps(
    total_samples: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    total_epochs: int,
) -> int:
    """
    Calculate total training steps from dataset size and training config.

    Args:
        total_samples: Total number of samples in dataset
        batch_size: Batch size
        gradient_accumulation_steps: Number of gradient accumulation steps
        total_epochs: Number of training epochs

    Returns:
        Total number of training steps
    """
    steps_per_epoch = total_samples // (batch_size * gradient_accumulation_steps)
    return steps_per_epoch * total_epochs


def create_dataset(
    config: DiaConfig,
    parquet_path: str | Path,
    max_audio_length: int | None = None,
    max_text_length: int | None = None,
    transcription_clean_prob: float = 0.1,
    text_dropout_prob: float = 0.15,
) -> AudioTextDataset:
    """
    Create an AudioTextDataset for training.

    Args:
        config: DiaConfig object
        parquet_path: Path to the parquet file
        max_audio_length: Maximum audio sequence length
        max_text_length: Maximum text sequence length
        transcription_clean_prob: Probability of using transcription_clean instead of transcription
        text_dropout_prob: Probability of dropping out text for CFG

    Returns:
        AudioTextDataset instance
    """
    return AudioTextDataset(
        parquet_path=parquet_path,
        config=config,
        max_audio_length=max_audio_length,
        max_text_length=max_text_length,
        transcription_clean_prob=transcription_clean_prob,
        text_dropout_prob=text_dropout_prob,
    )
