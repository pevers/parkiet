import jax.numpy as jnp
from dataclasses import dataclass
import flax.nnx as nnx
from parkiet.jax.state import create_attn_mask

class KVCacheTraining(nnx.Module):
    """
    JAX-friendly KV cache for training (smaller batch size, no CFG)
    """

    def __init__(
        self,
        batch_size: int,
        num_heads: int,
        max_len: int,
        head_dim: int,
        dtype: jnp.dtype,
        k: jnp.ndarray | None = None,
        v: jnp.ndarray | None = None,
        *,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ):
        # Training uses single batch size (no CFG doubling)
        shape = (batch_size, max_len, num_heads, head_dim)
        if k is None:
            k = jnp.zeros(shape, dtype=dtype)
        if v is None:
            v = jnp.zeros(shape, dtype=dtype)
        self.k = nnx.Variable(k, rngs=rngs)
        self.v = nnx.Variable(v, rngs=rngs)

    @classmethod
    def create(
        cls,
        batch_size: int,
        num_heads: int,
        max_len: int,
        head_dim: int,
        dtype: jnp.dtype,
        *,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ) -> "KVCacheTraining":
        return cls(batch_size, num_heads, max_len, head_dim, dtype, rngs=rngs)

    @classmethod
    def from_kv(
        cls, k: jnp.ndarray, v: jnp.ndarray, *, rngs: nnx.Rngs = nnx.Rngs(0)
    ) -> "KVCacheTraining":
        batch_size = k.shape[0]
        max_len = k.shape[1]
        num_heads = k.shape[2]
        head_dim = k.shape[3]
        dtype = k.dtype
        return cls(batch_size, num_heads, max_len, head_dim, dtype, k=k, v=v, rngs=rngs)

    def update(
        self,
        k: jnp.ndarray,  # [B, T, num_heads, head_dim]
        v: jnp.ndarray,  # same shape
        current_idx: int,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        self.k.value = self.k.value.at[:, current_idx, :, :].set(k.squeeze(1))
        self.v.value = self.v.value.at[:, current_idx, :, :].set(v.squeeze(1))
        return self.k.value, self.v.value

    def prefill(self, k: jnp.ndarray, v: jnp.ndarray):
        length = k.shape[1]
        self.k.value = self.k.value.at[:, :length, :, :].set(k)
        self.v.value = self.v.value.at[:, :length, :, :].set(v)


@dataclass
class EncoderTrainingState:
    """Parameters specifically for encoder training."""

    max_seq_len: int
    positions: jnp.ndarray  # [1, T]
    padding_mask: jnp.ndarray  # [B, T], dtype=bool
    attn_mask: jnp.ndarray  # [B, 1, T, T], dtype=bool

    @classmethod
    def new(cls, max_position_embeddings: int, src_tokens: jnp.ndarray) -> "EncoderTrainingState":
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
    self_attn_cache: list[KVCacheTraining]
    cross_attn_cache: list[KVCacheTraining]
    causal_attn_mask: jnp.ndarray  # [Tdec, Tdec]
    cross_attn_mask: jnp.ndarray  # [B, 1, Tdec, Tenc]

    @classmethod
    def new(
        cls,
        jax_config,  # Using the JaxConfig from train.py
        enc_state: EncoderTrainingState,
        enc_out: jnp.ndarray,
        dec_cross_attn_cache: list[KVCacheTraining],
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
        max_audio_len = max_generation_length or jax_config.decoder_max_position_embeddings
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
            KVCacheTraining(
                batch_size,
                jax_config.decoder_num_key_value_heads,
                max_audio_len,
                jax_config.decoder_cross_head_dim,
                compute_dtype,
            )
            for _ in range(jax_config.decoder_num_hidden_layers)
        ]

        return cls(
            dtype=compute_dtype,
            enc_out=enc_out,
            enc_positions=enc_state.positions,
            dec_positions=dec_positions,
            self_attn_cache=self_attn_cache,
            cross_attn_cache=dec_cross_attn_cache,
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


@dataclass
class TrainingDecoderOutput:
    generated_tokens: jnp.ndarray  # [B, Tdec, channels], dtype=int32
    prefill_steps: list[int]

    @classmethod
    def new(cls, batch_size: int, decoder_max_position_embeddings: int, decoder_num_channels: int) -> "TrainingDecoderOutput":
        audio_len = decoder_max_position_embeddings
        channels = decoder_num_channels
        tokens = jnp.full(
            (batch_size, audio_len, channels), fill_value=-1, dtype=jnp.int32
        )
        return cls(generated_tokens=tokens, prefill_steps=[])

    def get_tokens_at(self, step_from: int, step_to: int | None = None) -> jnp.ndarray:
        if step_to is None:
            step_to = step_from + 1
        return self.generated_tokens[:, step_from:step_to, :]

    def prefill(self, dec_out: jnp.ndarray, prefill_steps: list[int]) -> None:
        """
        dec_out: [B, L, channels], where L ≤ max audio length
        """
        length = dec_out.shape[1]
        self.generated_tokens = self.generated_tokens.at[:, :length, :].set(dec_out)
        self.prefill_steps = prefill_steps
