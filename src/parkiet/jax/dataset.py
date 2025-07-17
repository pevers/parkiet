"""
Dummy data loader for JAX training.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Iterator
from parkiet.dia.config import DiaConfig


class DummyDataLoader:
    """Dummy data loader that generates synthetic data for testing JAX training."""

    def __init__(
        self,
        config: DiaConfig,
        batch_size: int = 8,
        num_batches: int = 1000,
        seed: int = 42,
    ):
        """
        Initialize the dummy data loader.

        Args:
            config: Model configuration
            batch_size: Batch size for training
            num_batches: Number of batches to generate
            seed: Random seed for reproducibility
        """
        self.config = config
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        # Configuration parameters
        self.text_length = config.encoder_config.max_position_embeddings
        self.audio_length = config.decoder_config.max_position_embeddings
        self.num_channels = config.decoder_config.num_channels
        self.vocab_size = config.decoder_config.vocab_size

    def __iter__(self) -> Iterator[dict[str, jnp.ndarray]]:
        """Iterate over dummy batches."""
        for _ in range(self.num_batches):
            yield self._generate_batch()

    def __len__(self) -> int:
        """Return the number of batches."""
        return self.num_batches

    def _generate_batch(self) -> dict[str, jnp.ndarray]:
        """Generate a single batch of dummy data."""
        # Generate text tokens (byte-level encoding, 0-255)
        text_tokens = self.rng.randint(
            0, 256, size=(self.batch_size, self.text_length), dtype=np.int32
        )

        # Generate audio tokens (codebook indices)
        audio_tokens = self.rng.randint(
            0,
            self.vocab_size,
            size=(self.batch_size, self.audio_length, self.num_channels),
            dtype=np.int32,
        )

        # Convert to JAX arrays
        return {
            "text": jnp.array(text_tokens),
            "audio": jnp.array(audio_tokens),
        }


def create_dummy_dataloader(
    config: DiaConfig,
    batch_size: int = 8,
    num_batches: int = 1000,
    seed: int = 42,
) -> DummyDataLoader:
    """
    Create a dummy data loader for testing.

    Args:
        config: Model configuration
        batch_size: Batch size for training
        num_batches: Number of batches to generate
        seed: Random seed for reproducibility

    Returns:
        DummyDataLoader instance
    """
    return DummyDataLoader(config, batch_size, num_batches, seed)
