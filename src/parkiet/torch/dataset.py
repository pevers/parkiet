import numpy as np
import pandas as pd
from pathlib import Path
import logging
import random
from collections.abc import Iterator
from parkiet.dia.config import DiaConfig
from parkiet.jax.audio import build_delay_indices, apply_audio_delay

log = logging.getLogger(__name__)


class AudioTextDataset:
    """JAX dataset for loading audio-text pairs from parquet files created by the chunker."""

    def __init__(
        self,
        parquet_path: str | Path,
        config: DiaConfig,
        max_audio_length: int | None = None,
        max_text_length: int | None = None,
        transcript_clean_probability: float = 0.15,
        text_dropout_probability: float = 0.15,
    ):
        """
        Initialize the dataset.

        Args:
            parquet_path: Path to the parquet file containing processed chunks
            config: DiaConfig object containing model configuration
            max_audio_length: Maximum audio sequence length (in tokens). If None, uses config value.
            max_text_length: Maximum text sequence length (in tokens). If None, uses config value.
            transcript_clean_probability: Probability of using transcript_clean instead of transcription (default 0.15)
            text_dropout_probability: Probability of dropping text condition for classifier-free guidance (default 0.15)
        """
        self.parquet_path = Path(parquet_path)
        self.config = config
        self.max_audio_length = (
            max_audio_length or config.decoder_config.max_position_embeddings
        )
        self.max_text_length = (
            max_text_length or config.encoder_config.max_position_embeddings
        )
        self.transcript_clean_probability = transcript_clean_probability
        self.text_dropout_probability = text_dropout_probability

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
            & (self.df["speaker_weight"].notna())
        ].reset_index(drop=True)

        log.info(f"Loaded {len(self.df)} valid samples from dataset")

        # Extract speaker weights for weighted sampling
        self.sample_weights = self.df["speaker_weight"].values
        # Add small epsilon to avoid zero weights
        self.sample_weights = self.sample_weights + 1e-8
        # Normalize weights to sum to 1
        self.sample_weights = self.sample_weights / self.sample_weights.sum()

        log.info(
            f"Speaker weights range: {self.sample_weights.min():.4f} to {self.sample_weights.max():.4f}"
        )

        # Get data configuration values
        self.text_length = config.encoder_config.max_position_embeddings
        self.audio_pad_value = config.pad_token_id
        self.channels = config.decoder_config.num_channels

        # Store token IDs for preprocessing
        self.bos_token_id = config.bos_token_id
        self.eos_token_id = config.eos_token_id
        self.pad_token_id = config.pad_token_id
        self.delay_pattern = config.delay_pattern

    def _prepare_audio_sequence(
        self, audio_tokens: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Prepare audio sequence with BOS/EOS tokens and delay patterns.

        Args:
            audio_tokens: Raw audio tokens [T, C]

        Returns:
            Tuple of (input_sequence, target_sequence) both with shape [T+1, C]
        """
        # First add BOS/EOS tokens to the raw audio
        # Add BOS token at the beginning: [BOS, audio_0, audio_1, ..., audio_T-1]
        bos_tokens = np.full(
            (1, self.channels), self.bos_token_id, dtype=audio_tokens.dtype
        )
        input_sequence = np.concatenate([bos_tokens, audio_tokens], axis=0)  # [T+1, C]

        # Create target sequence: [audio_0, audio_1, ..., audio_T-1, EOS]
        target_sequence = np.concatenate(
            [
                audio_tokens,
                np.full(
                    (1, self.channels), self.eos_token_id, dtype=audio_tokens.dtype
                ),
            ],
            axis=0,
        )  # [T+1, C]

        # Now pad or truncate to max length
        current_len = input_sequence.shape[0]  # T+1
        if current_len > self.max_audio_length:
            input_sequence = input_sequence[: self.max_audio_length, :]
            target_sequence = target_sequence[: self.max_audio_length, :]
        elif current_len < self.max_audio_length:
            padding = np.full(
                (self.max_audio_length - current_len, self.channels),
                self.pad_token_id,
                dtype=audio_tokens.dtype,
            )
            input_sequence = np.concatenate([input_sequence, padding], axis=0)
            target_sequence = np.concatenate([target_sequence, padding], axis=0)

        # Apply delay pattern to both input and target
        batch_size = 1  # Single sample
        seq_len = input_sequence.shape[0]  # max_audio_length

        # Build delay indices once
        delay_indices = build_delay_indices(
            batch_size, seq_len, self.channels, self.delay_pattern
        )

        # Apply delay to input sequence
        input_delayed = apply_audio_delay(
            input_sequence[None, :, :],  # Add batch dimension [1, max_len, C]
            pad_value=-1,
            bos_value=self.bos_token_id,
            precomp=delay_indices,
        )[0]  # Remove batch dimension back to [max_len, C]

        # Apply delay to target sequence
        target_delayed = apply_audio_delay(
            target_sequence[None, :, :],  # Add batch dimension [1, max_len, C]
            pad_value=-1,
            bos_value=self.bos_token_id,
            precomp=delay_indices,
        )[0]  # Remove batch dimension back to [max_len, C]

        return input_delayed, target_delayed

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        """
        Get a single sample from the dataset.

        Args:
            idx: Index of the sample

        Returns:
            Dictionary containing 'text', 'audio_input', and 'audio_target' numpy arrays
        """
        row = self.df.iloc[idx]

        # Choose between transcription and transcription_clean based on probability
        use_clean = random.random() < self.transcript_clean_probability
        transcript_text = (
            row["transcription_clean"]
            if use_clean
            and pd.notna(row["transcription_clean"])
            and len(row["transcription_clean"]) > 0
            else row["transcription"]
        )

        # Process text
        text_tokens = self._encode_text(transcript_text)

        # Apply text dropout for classifier-free guidance
        if random.random() < self.text_dropout_probability:
            text_tokens.fill(0)

        # Process audio
        audio_tokens = self._decode_audio(
            row["encoded_audio"], row["encoded_audio_shape"]
        )

        # Prepare input and target sequences with delay patterns
        audio_input, audio_target = self._prepare_audio_sequence(audio_tokens)

        return {
            "text": text_tokens,
            "audio_input": audio_input,  # [T+1, C] with BOS and delay pattern
            "audio_target": audio_target,  # [T+1, C] with EOS and delay pattern
        }

    def _encode_text(self, text: str) -> np.ndarray:
        """
        Encode text using byte-level encoding (same as DIA model).

        Args:
            text: Input text string

        Returns:
            Encoded text array of shape [max_text_length] (padded to full length)
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

        return np.array(text_tokens, dtype=np.int64)

    def _decode_audio(
        self, encoded_audio: np.ndarray, encoded_audio_shape: np.ndarray
    ) -> np.ndarray:
        """
        Decode flattened audio data back to its original tensor shape.

        Args:
            encoded_audio: Flattened audio data as list of integers
            encoded_audio_shape: Original shape of the audio tensor [T, C]

        Returns:
            Audio array of shape [max_audio_length, channels]
        """
        # Reconstruct the original array
        audio_array = np.array(encoded_audio, dtype=np.int64).reshape(
            encoded_audio_shape.tolist()
        )

        # Ensure it has the expected number of channels
        if audio_array.shape[-1] != self.channels:
            raise ValueError(
                f"Audio array has {audio_array.shape[-1]} channels, "
                f"but config expects {self.channels} channels"
            )

        return audio_array

    def get_sample_weights(self) -> np.ndarray:
        """
        Get sample weights for weighted sampling.

        Returns:
            Array of normalized sample weights
        """
        return self.sample_weights

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
            "transcription_clean": row.get("transcription_clean", ""),
            "speaker_weight": row["speaker_weight"],
            "original_audio_shape": row["encoded_audio_shape"],
        }

    def batch_iterator(
        self, batch_size: int, shuffle: bool = True, use_weighted_sampling: bool = True
    ) -> Iterator[dict[str, np.ndarray]]:
        """
        Create a batch iterator for JAX training.

        Args:
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle indices (ignored if using weighted sampling)
            use_weighted_sampling: Whether to use speaker weight-based sampling

        Yields:
            Batches of data as dictionaries with 'text', 'audio_input', and 'audio_target' arrays
        """
        num_samples = len(self)

        while True:  # Infinite iterator
            if use_weighted_sampling:
                # Sample indices based on weights
                indices = np.random.choice(
                    num_samples, size=num_samples, replace=True, p=self.sample_weights
                )
            else:
                indices = np.arange(num_samples)
                if shuffle:
                    np.random.shuffle(indices)

            # Create batches
            for i in range(0, num_samples, batch_size):
                batch_indices = indices[i : i + batch_size]
                if len(batch_indices) < batch_size:
                    continue  # Skip incomplete batches

                batch_data = {"text": [], "audio_input": [], "audio_target": []}

                for idx in batch_indices:
                    sample = self[idx]
                    batch_data["text"].append(sample["text"])
                    batch_data["audio_input"].append(sample["audio_input"])
                    batch_data["audio_target"].append(sample["audio_target"])

                # Convert to numpy arrays and stack
                batch_data["text"] = np.stack(batch_data["text"], axis=0)
                batch_data["audio_input"] = np.stack(batch_data["audio_input"], axis=0)
                batch_data["audio_target"] = np.stack(
                    batch_data["audio_target"], axis=0
                )

                yield batch_data


def create_dataset(
    parquet_path: str | Path,
    config: DiaConfig,
    max_audio_length: int | None = None,
    max_text_length: int | None = None,
    # TODO: CHANGE BACK WHEN TRAINING FR
    transcript_clean_probability: float = 0.0,
    text_dropout_probability: float = 0.0,
) -> AudioTextDataset:
    """
    Create an AudioTextDataset for JAX training.

    Args:
        parquet_path: Path to the parquet file
        config: DiaConfig object
        max_audio_length: Maximum audio sequence length
        max_text_length: Maximum text sequence length
        transcript_clean_probability: Probability of using transcript_clean (default 0.15)
        text_dropout_probability: Probability of dropping text condition for classifier-free guidance (default 0.15)

    Returns:
        AudioTextDataset instance
    """
    return AudioTextDataset(
        parquet_path=parquet_path,
        config=config,
        max_audio_length=max_audio_length,
        max_text_length=max_text_length,
        transcript_clean_probability=transcript_clean_probability,
        text_dropout_probability=text_dropout_probability,
    )
