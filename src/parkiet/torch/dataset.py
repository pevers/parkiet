import numpy as np
import pandas as pd
from pathlib import Path
import logging
import random
from collections.abc import Iterator
from parkiet.dia.config import DiaConfig
from parkiet.jax.audio import build_delay_indices, apply_audio_delay
from parkiet.storage.gcs_client import GCSClient
import pyarrow.parquet as pq
import io
import json
import hashlib

log = logging.getLogger(__name__)


class AudioTextDataset:
    """JAX dataset for loading audio-text pairs from parquet files/shards created by the chunker."""

    def __init__(
        self,
        parquet_path: str | Path | None = None,
        config: DiaConfig | None = None,
        max_audio_length: int | None = None,
        max_text_length: int | None = None,
        transcript_clean_probability: float = 0.15,
        text_dropout_probability: float = 0.15,
        gcs_bucket: str | None = None,
        gcs_prefix: str | None = None,
    ):
        """
        Initialize the dataset.

        Args:
            parquet_path: Path to local parquet file (for single file mode)
            config: DiaConfig object containing model configuration
            max_audio_length: Maximum audio sequence length (in tokens). If None, uses config value.
            max_text_length: Maximum text sequence length (in tokens). If None, uses config value.
            transcript_clean_probability: Probability of using transcript_clean instead of transcription (default 0.15)
            text_dropout_probability: Probability of dropping text condition for classifier-free guidance (default 0.15)
            gcs_bucket: GCS bucket name (for GCS shards mode)
            gcs_prefix: GCS prefix for shard files (for GCS shards mode)
        """
        self.config = config
        self.max_audio_length = (
            max_audio_length or config.decoder_config.max_position_embeddings
        )
        self.max_text_length = (
            max_text_length or config.encoder_config.max_position_embeddings
        )
        self.transcript_clean_probability = transcript_clean_probability
        self.text_dropout_probability = text_dropout_probability

        # Determine loading mode
        self.use_gcs = gcs_bucket is not None and gcs_prefix is not None

        if self.use_gcs:
            # GCS shards mode - discover shards but don't load data yet
            self.gcs_bucket = gcs_bucket
            self.gcs_prefix = gcs_prefix
            log.info(f"Discovering shards in GCS: {gcs_bucket}/{gcs_prefix}")
            self._discover_gcs_shards()
            self.df = None  # No single DataFrame in shard mode
        else:
            # Local file mode
            if parquet_path is None:
                raise ValueError(
                    "Must provide either parquet_path or (gcs_bucket + gcs_prefix)"
                )
            self.parquet_path = Path(parquet_path)
            if not self.parquet_path.exists():
                raise FileNotFoundError(f"Parquet file not found: {self.parquet_path}")
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

        # Get data configuration values
        self.text_length = config.encoder_config.max_position_embeddings
        self.audio_pad_value = config.pad_token_id
        self.channels = config.decoder_config.num_channels

        # Store token IDs for preprocessing
        self.bos_token_id = config.bos_token_id
        self.eos_token_id = config.eos_token_id
        self.pad_token_id = config.pad_token_id
        self.delay_pattern = config.delay_pattern

    def _get_cache_path(self) -> Path:
        """Generate cache file path based on GCS bucket and prefix."""
        # Create a hash of the GCS path for a unique cache filename
        cache_key = f"{self.gcs_bucket}/{self.gcs_prefix}"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:8]
        cache_dir = Path.home() / ".cache" / "parkiet"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"shard_metadata_{cache_hash}.json"

    def _load_cached_metadata(self) -> bool:
        """
        Try to load cached shard metadata.

        Returns:
            True if cache was loaded successfully, False otherwise
        """
        cache_path = self._get_cache_path()
        if not cache_path.exists():
            return False

        try:
            with open(cache_path, "r") as f:
                cache_data = json.load(f)

            # Validate cache format
            if not all(
                key in cache_data
                for key in ["bucket", "prefix", "shards", "total_samples"]
            ):
                log.warning("Invalid cache format, rebuilding metadata")
                return False

            # Check if cache matches current GCS path
            if (
                cache_data["bucket"] != self.gcs_bucket
                or cache_data["prefix"] != self.gcs_prefix
            ):
                return False

            # Rebuild data structures from cache
            self.shard_info = []
            self.shard_index_map = {}
            self.shard_cache = {}
            self.total_samples = cache_data["total_samples"]

            gcs_client = GCSClient()

            for shard_data in cache_data["shards"]:
                blob_name = shard_data["blob_name"]
                blob = gcs_client.bucket.blob(blob_name)

                shard_info = {
                    "blob_name": blob_name,
                    "blob": blob,
                    "size": shard_data["size"],
                    "start_idx": shard_data["start_idx"],
                    "end_idx": shard_data["end_idx"],
                }
                self.shard_info.append(shard_info)

                # Rebuild index mapping
                for local_idx in range(shard_data["size"]):
                    global_idx = shard_data["start_idx"] + local_idx
                    shard_idx = len(self.shard_info) - 1
                    self.shard_index_map[global_idx] = (shard_idx, local_idx)

            log.info(
                f"Loaded shard metadata from cache: {len(self.shard_info)} shards, {self.total_samples} samples"
            )
            return True

        except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
            log.warning(f"Failed to load cache: {e}, rebuilding metadata")
            return False

    def _save_metadata_cache(self) -> None:
        """Save shard metadata to cache file."""
        cache_path = self._get_cache_path()

        # Prepare cache data (exclude blob objects which aren't JSON serializable)
        cache_data = {
            "bucket": self.gcs_bucket,
            "prefix": self.gcs_prefix,
            "total_samples": self.total_samples,
            "shards": [],
        }

        for shard_info in self.shard_info:
            cache_data["shards"].append(
                {
                    "blob_name": shard_info["blob_name"],
                    "size": shard_info["size"],
                    "start_idx": shard_info["start_idx"],
                    "end_idx": shard_info["end_idx"],
                }
            )

        try:
            with open(cache_path, "w") as f:
                json.dump(cache_data, f, indent=2)
            log.info(f"Saved shard metadata cache to: {cache_path}")
        except Exception as e:
            log.warning(f"Failed to save metadata cache: {e}")

    def _discover_gcs_shards(self) -> None:
        """
        Discover parquet shards in GCS and build index mapping.
        Uses cached metadata if available.
        """
        # Try to load from cache first
        if self._load_cached_metadata():
            return

        # Cache miss - discover shards from GCS
        log.info(f"Discovering shards in GCS: {self.gcs_bucket}/{self.gcs_prefix}")
        gcs_client = GCSClient()

        # List all parquet files with the given prefix
        blobs = list(gcs_client.bucket.list_blobs(prefix=self.gcs_prefix))
        parquet_blobs = [blob for blob in blobs if blob.name.endswith(".parquet")]

        if not parquet_blobs:
            raise FileNotFoundError(
                f"No parquet files found in GCS at {self.gcs_bucket}/{self.gcs_prefix}"
            )

        # Sort shards by name for consistent ordering
        parquet_blobs.sort(key=lambda b: b.name)

        log.info(f"Found {len(parquet_blobs)} parquet shards in GCS")

        # Build shard index mapping
        self.shard_info = []
        self.shard_index_map = {}  # global_idx -> (shard_idx, local_idx)
        self.shard_cache = {}  # shard_idx -> DataFrame
        self.total_samples = 0

        # Get sample counts from each shard (metadata only)
        for shard_idx, blob in enumerate(parquet_blobs):
            # Download to memory and read metadata
            blob_data = blob.download_as_bytes()

            # Read just the metadata/schema to get row count from memory
            parquet_file = pq.ParquetFile(io.BytesIO(blob_data))
            shard_size = parquet_file.metadata.num_rows

            shard_info = {
                "blob_name": blob.name,
                "blob": blob,
                "size": shard_size,
                "start_idx": self.total_samples,
                "end_idx": self.total_samples + shard_size,
            }
            self.shard_info.append(shard_info)

            # Build index mapping
            for local_idx in range(shard_size):
                global_idx = self.total_samples + local_idx
                self.shard_index_map[global_idx] = (shard_idx, local_idx)

            self.total_samples += shard_size
            log.info(f"  Shard {shard_idx}: {blob.name} ({shard_size} samples)")

        log.info(f"Total samples across all shards: {self.total_samples}")

        # Save metadata to cache for next time
        self._save_metadata_cache()

    def _load_shard(self, shard_idx: int) -> pd.DataFrame:
        """
        Load a specific shard and cache it.

        Args:
            shard_idx: Index of the shard to load

        Returns:
            DataFrame for the shard
        """
        if shard_idx in self.shard_cache:
            return self.shard_cache[shard_idx]

        shard_info = self.shard_info[shard_idx]
        blob = shard_info["blob"]

        log.debug(f"Loading shard {shard_idx}: {blob.name}")

        # Download to memory and read parquet directly from bytes
        blob_data = blob.download_as_bytes()
        shard_df = pd.read_parquet(io.BytesIO(blob_data))

        # Filter out invalid rows
        shard_df = shard_df[
            (shard_df["transcription"].notna())
            & (shard_df["transcription"].str.len() > 0)
            & (shard_df["encoded_audio"].notna())
            & (shard_df["speaker_weight"].notna())
        ].reset_index(drop=True)

        # Cache the shard (with LRU eviction if cache gets too large)
        if len(self.shard_cache) >= 3:  # Keep max 3 shards in memory
            # Remove oldest shard
            oldest_shard_idx = min(self.shard_cache.keys())
            del self.shard_cache[oldest_shard_idx]
            log.debug(f"Evicted shard {oldest_shard_idx} from cache")

        self.shard_cache[shard_idx] = shard_df
        return shard_df

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
        if self.use_gcs:
            return self.total_samples
        else:
            return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        """
        Get a single sample from the dataset.

        Args:
            idx: Index of the sample

        Returns:
            Dictionary containing 'text', 'audio_input', and 'audio_target' numpy arrays
        """
        if self.use_gcs:
            # GCS shard mode - load shard on demand
            if idx >= self.total_samples:
                raise IndexError(
                    f"Index {idx} out of range (total samples: {self.total_samples})"
                )

            shard_idx, local_idx = self.shard_index_map[idx]
            shard_df = self._load_shard(shard_idx)
            row = shard_df.iloc[local_idx]
        else:
            # Local file mode
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

    def get_sample_weight(self, idx: int) -> float:
        """
        Get sample weight for a specific sample.

        Args:
            idx: Index of the sample

        Returns:
            Sample weight from the dataset
        """
        if self.use_gcs:
            shard_idx, local_idx = self.shard_index_map[idx]
            shard_df = self._load_shard(shard_idx)
            return float(shard_df.iloc[local_idx]["speaker_weight"])
        else:
            return float(self.df.iloc[idx]["speaker_weight"])

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
                # Collect weights for all samples
                weights = np.array(
                    [self.get_sample_weight(i) for i in range(num_samples)]
                )
                # Normalize weights to sum to 1
                weights = weights / weights.sum()
                indices = np.random.choice(
                    num_samples, size=num_samples, replace=True, p=weights
                )
            else:
                # Sample uniformly
                indices = np.random.choice(num_samples, size=num_samples, replace=True)
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
    config: DiaConfig,
    parquet_path: str | Path | None = None,
    max_audio_length: int | None = None,
    max_text_length: int | None = None,
    transcript_clean_probability: float = 0.15,
    text_dropout_probability: float = 0.15,
    gcs_bucket: str | None = None,
    gcs_prefix: str | None = None,
) -> AudioTextDataset:
    """
    Create an AudioTextDataset for JAX training.

    Args:
        config: DiaConfig object
        parquet_path: Path to the parquet file (for local mode)
        max_audio_length: Maximum audio sequence length
        max_text_length: Maximum text sequence length
        transcript_clean_probability: Probability of using transcript_clean (default 0.15)
        text_dropout_probability: Probability of dropping text condition for classifier-free guidance (default 0.15)
        gcs_bucket: GCS bucket name (for GCS shards mode)
        gcs_prefix: GCS prefix for shard files (for GCS shards mode)

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
        gcs_bucket=gcs_bucket,
        gcs_prefix=gcs_prefix,
    )
