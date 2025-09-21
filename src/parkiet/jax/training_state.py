import jax.numpy as jnp
from dataclasses import dataclass
from parkiet.jax.state import create_attn_mask
from parkiet.jax.model import DiaConfig


@dataclass
class EncoderTrainingState:
    """Parameters specifically for encoder training."""

    max_seq_len: int
    positions: jnp.ndarray  # [1, T]
    padding_mask: jnp.ndarray  # [B, T], dtype=bool
    attn_mask: jnp.ndarray  # [B, 1, T, T], dtype=bool

    @classmethod
    def new(
        cls, max_position_embeddings: int, src_tokens: jnp.ndarray
    ) -> "EncoderTrainingState":
        """
        Create encoder training state.

        Args:
            max_position_embeddings: Maximum position embeddings
            src_tokens: [B, T] of token IDs

        Returns:
            EncoderTrainingState instance
        """
        seq_len = max_position_embeddings
        # [1, T]
        positions = jnp.arange(seq_len, dtype=jnp.float32)[None, :]
        # [B, T] mask non-padding
        padding_mask = src_tokens != 0
        # Self-attention mask
        attn_mask = create_attn_mask(padding_mask, padding_mask, is_causal=False)

        return cls(
            max_seq_len=seq_len,
            positions=positions,
            padding_mask=padding_mask,
            attn_mask=attn_mask,
        )


@dataclass
class DecoderTrainingState:
    """Parameters specifically for decoder training."""

    dtype: jnp.dtype
    enc_out: jnp.ndarray  # [B, Tenc, D]
    enc_positions: jnp.ndarray  # [1, Tenc]
    dec_positions: jnp.ndarray  # [1, Tdec]
    self_attn_cache: list[None]
    cross_attn_cache: list[None]
    causal_attn_mask: jnp.ndarray  # [Tdec, Tdec]
    cross_attn_mask: jnp.ndarray  # [B, 1, Tdec, Tenc]

    @classmethod
    def new(
        cls,
        dia_config: DiaConfig,
        enc_state: EncoderTrainingState,
        enc_out: jnp.ndarray,
        compute_dtype: jnp.dtype,
        max_generation_length: int | None = None,
    ) -> "DecoderTrainingState":
        """
        Create decoder training state.

        Args:
            jax_config: JAX configuration object
            enc_state: Encoder training state
            enc_out: Encoder outputs [B, Tenc, D]
            dec_cross_attn_cache: Cross-attention cache
            compute_dtype: Computation dtype
            max_generation_length: Maximum generation length

        Returns:
            DecoderTrainingState instance
        """
        max_audio_len = (
            max_generation_length or dia_config.decoder_config.max_position_embeddings
        )
        batch_size = enc_out.shape[0]

        # Create decoder positions for the sequence length
        dec_positions = jnp.arange(max_audio_len, dtype=jnp.int32)[None, :]

        # Create causal mask for self-attention
        causal_mask = jnp.tril(jnp.ones((max_audio_len, max_audio_len), dtype=bool))

        # Create cross-attention mask
        dec_mask = jnp.ones((batch_size, max_audio_len), dtype=jnp.bool_)
        cross_attn_mask = create_attn_mask(
            dec_mask, enc_state.padding_mask, is_causal=False
        )
        self_attn_cache = [
            None for _ in range(dia_config.decoder_config.num_hidden_layers)
        ]
        cross_attn_cache = [
            None for _ in range(dia_config.decoder_config.num_hidden_layers)
        ]

        return cls(
            dtype=compute_dtype,
            enc_out=enc_out,
            enc_positions=enc_state.positions,
            dec_positions=dec_positions,
            self_attn_cache=self_attn_cache,
            cross_attn_cache=cross_attn_cache,
            causal_attn_mask=causal_mask,
            cross_attn_mask=cross_attn_mask,
        )

    def prepare_step(self, step_from: int, step_to: int | None = None) -> None:
        """
        Update dec_positions for the next generation step(s).
        """
        if step_to is None:
            step_to = step_from + 1
        # shapes [1, step_to-step_from]
        self.dec_positions = jnp.arange(step_from, step_to, dtype=jnp.int32)[None, :]
