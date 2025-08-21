import jax.numpy as jnp
from dataclasses import dataclass
from parkiet.dia.config import DiaConfig
import flax.nnx as nnx


def create_attn_mask(
    q_padding_mask_1d: jnp.ndarray,  # shape [B, Tq], dtype=bool
    k_padding_mask_1d: jnp.ndarray,  # shape [B, Tk], dtype=bool
    is_causal: bool = False,
) -> jnp.ndarray:
    """
    Creates the attention mask (self or cross)
    Returns mask of shape [B, 1, Tq, Tk], dtype=bool.
    """
    # [B, Tq, 1] & [B, 1, Tk]
    p_mask_q = jnp.expand_dims(q_padding_mask_1d, axis=2)
    p_mask_k = jnp.expand_dims(k_padding_mask_1d, axis=1)

    # Condition A: Non-padding query attends to non-padding key
    non_pad_attends_non_pad = p_mask_q & p_mask_k

    # Condition B: Padding query attends to padding key
    pad_attends_pad = (~p_mask_q) & (~p_mask_k)

    # Combine: True if padding status is compatible (both non-pad OR both pad)
    mask = non_pad_attends_non_pad | pad_attends_pad  # Shape [B, Tq, Tk]

    if is_causal:
        # assume Tq == Tk
        causal_2d = jnp.tril(jnp.ones(mask.shape[-2:], dtype=bool))
        mask = mask & causal_2d

    return jnp.expand_dims(mask, axis=1)  # [B, 1, Tq, Tk]


@dataclass
class EncoderInferenceState:
    """Parameters specifically for encoder inference."""

    max_seq_len: int
    positions: jnp.ndarray  # [1, T]
    padding_mask: jnp.ndarray  # [2B, T], dtype=bool
    attn_mask: jnp.ndarray  # [2B, 1, T, T], dtype=bool

    @classmethod
    def new(cls, config: DiaConfig, cond_src: jnp.ndarray) -> "EncoderInferenceState":
        """
        cond_src: [B, 1, T] of token IDs
        """
        seq_len = config.encoder_config.max_position_embeddings
        # [1, T]
        positions = jnp.arange(seq_len, dtype=jnp.float32)[None, :]
        # [B, T] mask non-padding
        pad = cond_src.squeeze(1) != 0
        # duplicate each batch entry twice -> [2B, T]
        padding_mask = jnp.repeat(pad, repeats=2, axis=0)
        attn_mask = create_attn_mask(padding_mask, padding_mask, is_causal=False)

        return cls(
            max_seq_len=seq_len,
            positions=positions,
            padding_mask=padding_mask,
            attn_mask=attn_mask,
        )


class KVCache(nnx.Module):
    """
    JAX-friendly KV cache
    NOTE: Different shape for JAX because of order of dimensions for the attention layers
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
        # Different shape for JAX because of order of dimensions
        shape = (2 * batch_size, max_len, num_heads, head_dim)
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
    ) -> "KVCache":
        return cls(batch_size, num_heads, max_len, head_dim, dtype, rngs=rngs)

    @classmethod
    def from_kv(
        cls, k: jnp.ndarray, v: jnp.ndarray, *, rngs: nnx.Rngs = nnx.Rngs(0)
    ) -> "KVCache":
        batch_size = k.shape[0] // 2
        max_len = k.shape[1]
        num_heads = k.shape[2]
        head_dim = k.shape[3]
        dtype = k.dtype
        return cls(batch_size, num_heads, max_len, head_dim, dtype, k=k, v=v, rngs=rngs)

    def update(
        self,
        k: jnp.ndarray,  # [2B, T, num_heads, head_dim]
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
class DecoderInferenceState:
    """Parameters specifically for decoder inference."""

    dtype: jnp.dtype
    enc_out: jnp.ndarray  # [2B, Tenc, D]
    enc_positions: jnp.ndarray  # [1, Tenc]
    dec_positions: jnp.ndarray  # [1, Tdec_so_far]
    self_attn_cache: list[KVCache]
    cross_attn_cache: list[KVCache]
    causal_attn_mask: jnp.ndarray  # [Tdec, Tdec]
    cross_attn_mask: jnp.ndarray  # [2B, 1, Tdec, Tenc]

    @classmethod
    def new(
        cls,
        config: DiaConfig,
        enc_state: EncoderInferenceState,
        enc_out: jnp.ndarray,
        dec_cross_attn_cache: list[KVCache],
        compute_dtype: jnp.dtype,
        max_generation_length: int | None = None,
    ) -> "DecoderInferenceState":
        max_audio_len = (
            max_generation_length or config.decoder_config.max_position_embeddings
        )
        batch_size = enc_out.shape[0] // 2

        dec_positions = jnp.full((2 * batch_size, 1), fill_value=0, dtype=jnp.int32)
        causal_mask = jnp.tril(jnp.ones((max_audio_len, max_audio_len), dtype=bool))
        dec_mask = jnp.ones((2 * batch_size, 1), dtype=jnp.bool)
        cross_attn_mask = create_attn_mask(
            dec_mask, enc_state.padding_mask, is_causal=False
        )

        # instantiate fresh KV caches for self-attention
        self_caches = [
            KVCache.create(
                batch_size,
                config.decoder_config.num_key_value_heads,
                max_audio_len,
                config.decoder_config.head_dim,
                compute_dtype,
            )
            for _ in range(config.decoder_config.num_hidden_layers)
        ]

        return cls(
            dtype=compute_dtype,
            enc_out=enc_out,
            enc_positions=enc_state.positions,
            dec_positions=dec_positions,
            self_attn_cache=self_caches,
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
class DecoderOutput:
    generated_tokens: jnp.ndarray  # [B, Tdec, channels], dtype=int32
    prefill_steps: list[int]

    @classmethod
    def new(cls, batch_size: int, config: DiaConfig) -> "DecoderOutput":
        audio_len = config.decoder_config.max_position_embeddings
        channels = config.decoder_config.num_channels
        tokens = jnp.full(
            (batch_size, audio_len, channels), fill_value=-1, dtype=jnp.int32
        )
        return cls(generated_tokens=tokens, prefill_steps=[])

    def get_tokens_at(self, step_from: int, step_to: int | None = None) -> jnp.ndarray:
        if step_to is None:
            step_to = step_from + 1
        return self.generated_tokens[:, step_from:step_to, :]

    def update_one(
        self, dec_out: jnp.ndarray, step: int, apply_mask: bool = False
    ) -> None:
        """
        dec_out: [B, channels]
        """
        dec_out = dec_out.astype(self.generated_tokens.dtype)
        if apply_mask:
            mask = self.generated_tokens[:, step, :] == -1
            new_vals = jnp.where(mask, dec_out, self.generated_tokens[:, step, :])
        else:
            new_vals = dec_out
        self.generated_tokens = self.generated_tokens.at[:, step, :].set(new_vals)

    def prefill(self, dec_out: jnp.ndarray, prefill_steps: list[int]) -> None:
        """
        dec_out: [B, L, channels], where L ≤ max audio length
        """
        length = dec_out.shape[1]
        self.generated_tokens = self.generated_tokens.at[:, :length, :].set(dec_out)
        self.prefill_steps = prefill_steps
