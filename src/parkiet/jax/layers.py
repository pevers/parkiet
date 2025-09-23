import jax
import jax.numpy as jnp
import flax.nnx as nnx
from parkiet.dia.config import DecoderConfig, DiaConfig, EncoderConfig
from parkiet.jax.state import DecoderInferenceState, EncoderInferenceState, KVCache
from jax.nn import dot_product_attention


class MlpBlock(nnx.Module):
    """MLP block using nnx.Linear"""

    embed_dim: int
    intermediate_dim: int
    compute_dtype: jnp.dtype = jnp.float32

    def __init__(
        self,
        embed_dim: int,
        intermediate_dim: int,
        compute_dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ):
        ki = nnx.initializers.variance_scaling(1.0, "fan_in", "uniform")

        self.dtype = compute_dtype
        self.wi_fused = nnx.LinearGeneral(
            axis=(-1,),
            in_features=(embed_dim,),
            out_features=(2, intermediate_dim),
            use_bias=False,
            param_dtype=param_dtype,
            dtype=compute_dtype,
            # kernel_init=nnx.with_partitioning(ki, (None, None, "model")),
            rngs=rngs,
        )

        self.wo = nnx.LinearGeneral(
            axis=(-1,),
            in_features=(intermediate_dim,),
            out_features=(embed_dim,),
            use_bias=False,
            param_dtype=param_dtype,
            dtype=compute_dtype,
            # kernel_init=nnx.with_partitioning(ki, ("model", None)),
            rngs=rngs,
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass."""
        fused_x = self.wi_fused(x)
        gate = fused_x[..., 0, :]
        up = fused_x[..., 1, :]
        hidden = jax.nn.silu(gate) * up
        output = self.wo(hidden)
        return output


class RotaryEmbedding(nnx.Module):
    """Rotary Position Embedding (RoPE) implementation in JAX using nnx."""

    embedding_dims: int
    min_timescale: float = (1.0,)
    max_timescale: float = (10000.0,)
    compute_dtype: jnp.dtype = jnp.float32

    def __init__(
        self,
        embedding_dims: int,
        min_timescale: int = 1,
        max_timescale: int = 10000,
        compute_dtype: jnp.dtype = jnp.float32,
    ):
        if embedding_dims % 2 != 0:
            raise ValueError("Embedding dim must be even for RoPE.")
        self.embedding_dims = embedding_dims
        self.min_timescale = min_timescale
        self.max_timescale = max_timescale
        self.compute_dtype = compute_dtype

        half_embedding_dim = embedding_dims // 2
        fraction = (2.0 * jnp.arange(0, half_embedding_dim)) / embedding_dims
        timescale = (
            min_timescale * (max_timescale / min_timescale) ** fraction
        ).astype(jnp.float32)
        self.timescale = nnx.Variable(timescale)

    def __call__(self, inputs: jnp.ndarray, position: jnp.ndarray) -> jnp.ndarray:
        """Applies RoPE."""
        position = position[..., None, None]
        sinusoid_inp = position / self.timescale.value
        sin = jnp.sin(sinusoid_inp)
        cos = jnp.cos(sinusoid_inp)

        inputs_float = inputs.astype(jnp.float32)
        first_half, second_half = jnp.split(inputs_float, 2, axis=-1)
        first_part = first_half * cos - second_half * sin
        second_part = second_half * cos + first_half * sin

        return jnp.concatenate(
            (
                first_part.astype(self.compute_dtype),
                second_part.astype(self.compute_dtype),
            ),
            axis=-1,
        )

    def apply_rope(
        self, inputs: jnp.ndarray, sin: jnp.ndarray, cos: jnp.ndarray
    ) -> jnp.ndarray:
        """Apply precomputed sin/cos to inputs."""
        inputs_float = inputs.astype(jnp.float32)
        first_half, second_half = jnp.split(inputs_float, 2, axis=-1)
        first_part = first_half * cos - second_half * sin
        second_part = second_half * cos + first_half * sin
        return jnp.concatenate(
            (
                first_part.astype(self.compute_dtype),
                second_part.astype(self.compute_dtype),
            ),
            axis=-1,
        )


class CrossAttention(nnx.Module):
    """Cross-Attention using nnx.LinearGeneral."""

    config: EncoderConfig | DecoderConfig
    q_embed_dim: int
    kv_embed_dim: int
    num_query_heads: int
    num_kv_heads: int
    head_dim: int
    compute_dtype: jnp.dtype = jnp.float32
    out_embed_dim: int | None = None

    def __init__(
        self,
        config: EncoderConfig | DecoderConfig,
        q_embed_dim: int,
        kv_embed_dim: int,
        num_query_heads: int,
        num_kv_heads: int,
        head_dim: int,
        compute_dtype: jnp.dtype = jnp.float32,
        out_embed_dim: int | None = None,
        param_dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ):
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.output_dim = out_embed_dim if out_embed_dim is not None else q_embed_dim
        self.projected_query_dim = num_query_heads * head_dim
        if num_query_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_query_heads ({num_query_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
            )
        self.num_gqa_groups = num_query_heads // num_kv_heads

        ki = nnx.initializers.variance_scaling(1.0, "fan_in", "uniform")

        # Projection layers
        self.q_proj = nnx.LinearGeneral(
            axis=(-1,),
            in_features=(q_embed_dim,),
            out_features=(num_query_heads, head_dim),
            use_bias=False,
            dtype=compute_dtype,
            param_dtype=param_dtype,
            # kernel_init=nnx.with_partitioning(ki, (None, "model", None)),
            rngs=rngs,
        )
        self.k_proj = nnx.LinearGeneral(
            axis=(-1,),
            in_features=(kv_embed_dim,),
            out_features=(num_kv_heads, head_dim),
            use_bias=False,
            dtype=compute_dtype,
            param_dtype=param_dtype,
            # kernel_init=nnx.with_partitioning(ki, (None, "model", None)),
            rngs=rngs,
        )
        self.v_proj = nnx.LinearGeneral(
            axis=(-1,),
            in_features=(kv_embed_dim,),
            out_features=(num_kv_heads, head_dim),
            use_bias=False,
            dtype=compute_dtype,
            param_dtype=param_dtype,
            # kernel_init=nnx.with_partitioning(ki, (None, "model", None)),
            rngs=rngs,
        )
        self.o_proj = nnx.LinearGeneral(
            axis=(-2, -1),
            in_features=(num_query_heads, head_dim),
            out_features=(self.output_dim,),
            use_bias=False,
            dtype=compute_dtype,
            param_dtype=param_dtype,
            # kernel_init=nnx.with_partitioning(ki, ("model", None, None)),
            rngs=rngs,
        )

    def __call__(
        self,
        Xq: jnp.ndarray,  # (B, T, D) T = 1 in AR generation
        q_positions: jnp.ndarray,  # (B, T)
        kv_positions: jnp.ndarray | None = None,  # (B, S)
        attn_mask: jnp.ndarray
        | None = None,  # None in Decoder Self Attention, Valid mask in Others
        cache: KVCache | None = None,
        is_causal: bool = False,
        encoder_output: jnp.ndarray | None = None,  # (B, S, E) for cross-attention
    ) -> jnp.ndarray:
        if kv_positions is None:
            kv_positions = q_positions
        original_dtype = Xq.dtype

        Xq_BxNxTxH = self.q_proj(Xq)

        # During training (cache=None), compute keys/values from encoder output
        if cache is not None:
            attn_k, attn_v = cache.k, cache.v
        else:
            k = self.k_proj(encoder_output)
            v = self.v_proj(encoder_output)
            attn_k, attn_v = k, v

        attn_output = dot_product_attention(
            query=Xq_BxNxTxH,
            key=attn_k,
            value=attn_v,
            mask=attn_mask if is_causal else None,
            scale=1.0,
            is_causal=is_causal,
        )
        output = self.o_proj(attn_output)

        return output.astype(original_dtype)


class FusedQKV(nnx.Module):
    """Fused QKV projection for memory-efficient attention computation."""

    def __init__(
        self,
        in_features: int,
        num_q_heads: int,
        q_head_dim: int,
        num_kv_heads: int,
        kv_head_dim: int,
        bias: bool = False,
        compute_dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ):
        self.num_q_heads = num_q_heads
        self.q_head_dim = q_head_dim
        self.num_kv_heads = num_kv_heads
        self.kv_head_dim = kv_head_dim
        self.q_output_dim = num_q_heads * q_head_dim
        self.kv_output_dim = num_kv_heads * kv_head_dim
        self.total_output_dim = self.q_output_dim + 2 * self.kv_output_dim

        ki = nnx.initializers.variance_scaling(1.0, "fan_in", "uniform")
        self.linear = nnx.LinearGeneral(
            axis=(-1,),
            in_features=(in_features,),
            out_features=(self.total_output_dim,),
            use_bias=bias,
            dtype=compute_dtype,
            param_dtype=param_dtype,
            # kernel_init=nnx.with_partitioning(ki, (None, "model")),
            rngs=rngs,
        )

    def __call__(
        self, inputs: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        x = self.linear(inputs)

        # Split into Q, K, V projections
        q, k, v = jnp.split(
            x, [self.q_output_dim, self.q_output_dim + self.kv_output_dim], axis=-1
        )

        # Reshape to include head dimensions
        q = q.reshape(q.shape[:-1] + (self.num_q_heads, self.q_head_dim))
        k = k.reshape(k.shape[:-1] + (self.num_kv_heads, self.kv_head_dim))
        v = v.reshape(v.shape[:-1] + (self.num_kv_heads, self.kv_head_dim))

        return q, k, v


class SelfAttention(nnx.Module):
    """Self-Attention using nnx.LinearGeneral."""

    config: EncoderConfig | DecoderConfig
    q_embed_dim: int
    kv_embed_dim: int
    num_query_heads: int
    num_kv_heads: int
    head_dim: int
    compute_dtype: jnp.dtype = jnp.float32
    out_embed_dim: int | None = None

    def __init__(
        self,
        config: EncoderConfig | DecoderConfig,
        q_embed_dim: int,
        kv_embed_dim: int,
        num_query_heads: int,
        num_kv_heads: int,
        head_dim: int,
        compute_dtype: jnp.dtype = jnp.float32,
        out_embed_dim: int | None = None,
        param_dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ):
        ki = nnx.initializers.variance_scaling(1.0, "fan_in", "uniform")
        self.config = config
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.output_dim = out_embed_dim if out_embed_dim is not None else q_embed_dim
        self.projected_query_dim = num_query_heads * head_dim
        if num_query_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_query_heads ({num_query_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
            )
        self.num_gqa_groups = num_query_heads // num_kv_heads

        # Use FusedQKV for memory efficiency in self-attention
        # TODO: This is slow!?
        # if q_embed_dim == kv_embed_dim:
        #     self.use_fused_qkv = True
        #     self.qkv_proj = FusedQKV(
        #         in_features=q_embed_dim,
        #         num_q_heads=num_query_heads,
        #         q_head_dim=head_dim,
        #         num_kv_heads=num_kv_heads,
        #         kv_head_dim=head_dim,
        #         compute_dtype=compute_dtype,
        #         param_dtype=param_dtype,
        #         rngs=rngs,
        #     )
        # else:
        # Fallback to separate projections for cross-attention scenarios

        self.use_fused_qkv = False
        self.q_proj = nnx.LinearGeneral(
            in_features=(q_embed_dim,),
            out_features=(num_query_heads, head_dim),
            axis=(-1,),
            param_dtype=param_dtype,
            dtype=compute_dtype,
            use_bias=False,
            # kernel_init=nnx.with_partitioning(ki, (None, "model", None)),
            rngs=rngs,
        )
        self.k_proj = nnx.LinearGeneral(
            in_features=(kv_embed_dim,),
            out_features=(num_kv_heads, head_dim),
            axis=(-1,),
            param_dtype=param_dtype,
            dtype=compute_dtype,
            use_bias=False,
            # kernel_init=nnx.with_partitioning(ki, (None, "model", None)),
            rngs=rngs,
        )
        self.v_proj = nnx.LinearGeneral(
            in_features=(kv_embed_dim,),
            out_features=(num_kv_heads, head_dim),
            axis=(-1,),
            param_dtype=param_dtype,
            dtype=compute_dtype,
            use_bias=False,
            # kernel_init=nnx.with_partitioning(ki, (None, "model", None)),
            rngs=rngs,
        )

        self.o_proj = nnx.LinearGeneral(
            in_features=(num_query_heads, head_dim),
            out_features=(self.output_dim,),
            axis=(-2, -1),
            param_dtype=param_dtype,
            dtype=compute_dtype,
            use_bias=False,
            kernel_init=nnx.with_partitioning(ki, ("model", None, None)),
            rngs=rngs,
        )

        self.rotary_emb = RotaryEmbedding(
            embedding_dims=head_dim,
            max_timescale=config.rope_theta,
            compute_dtype=compute_dtype,
        )

    def __call__(
        self,
        X: jnp.ndarray,  # (B, T, D) T = 1 in AR generation
        q_positions: jnp.ndarray,  # (B, T)
        kv_positions: jnp.ndarray | None = None,  # (B, S)
        attn_mask: jnp.ndarray
        | None = None,  # None in Decoder Self Attention, Valid mask in Others
        cache: KVCache | None = None,
        prefill: bool = False,
        is_causal: bool = False,
        current_idx: int | None = None,
    ) -> jnp.ndarray:
        if kv_positions is None:
            kv_positions = q_positions

        original_dtype = X.dtype

        if self.use_fused_qkv:
            Xq_BxTxNxH, Xk_BxSxKxH, Xv_BxSxKxH = self.qkv_proj(X)
        else:
            Xq_BxTxNxH = self.q_proj(X)
            Xk_BxSxKxH = self.k_proj(X)
            Xv_BxSxKxH = self.v_proj(X)

        # Apply RoPE to queries
        q_position = q_positions[..., None, None]
        q_sinusoid_inp = q_position / self.rotary_emb.timescale.value
        q_sin = jnp.sin(q_sinusoid_inp)
        q_cos = jnp.cos(q_sinusoid_inp)

        # Apply RoPE to keys
        kv_position = kv_positions[..., None, None]
        kv_sinusoid_inp = kv_position / self.rotary_emb.timescale.value
        kv_sin = jnp.sin(kv_sinusoid_inp)
        kv_cos = jnp.cos(kv_sinusoid_inp)

        Xq_BxTxNxH = self.rotary_emb.apply_rope(Xq_BxTxNxH, q_sin, q_cos)
        Xk_BxSxKxH = self.rotary_emb.apply_rope(Xk_BxSxKxH, kv_sin, kv_cos)

        # During training (cache=None), use keys/values directly without caching
        if cache is None:
            attn_k = Xk_BxSxKxH
            attn_v = Xv_BxSxKxH
        elif prefill:
            attn_k, attn_v = Xk_BxSxKxH, Xv_BxSxKxH
            cache.prefill(attn_k, attn_v)
        else:
            attn_k, attn_v = cache.update(Xk_BxSxKxH, Xv_BxSxKxH, current_idx)

        attn_output = dot_product_attention(
            query=Xq_BxTxNxH,
            key=attn_k,
            value=attn_v,
            mask=attn_mask if not is_causal else None,
            scale=1.0,
            is_causal=is_causal,
        )

        output = self.o_proj(attn_output)
        return output.astype(original_dtype)


class EncoderLayer(nnx.Module):
    """Transformer Encoder Layer."""

    config: DiaConfig
    compute_dtype: jnp.dtype = jnp.float32

    def __init__(
        self,
        config: DiaConfig,
        compute_dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ):
        self.config = config
        enc_config = self.config.encoder_config
        embed_dim = enc_config.hidden_size
        self.compute_dtype = compute_dtype

        self.pre_sa_norm = nnx.RMSNorm(
            num_features=embed_dim,
            epsilon=enc_config.norm_eps,
            dtype=jnp.float32,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.self_attention = SelfAttention(
            config=self.config.encoder_config,
            q_embed_dim=embed_dim,
            kv_embed_dim=embed_dim,
            num_query_heads=enc_config.num_attention_heads,
            num_kv_heads=enc_config.num_key_value_heads,
            head_dim=enc_config.head_dim,
            compute_dtype=self.compute_dtype,
            param_dtype=param_dtype,
            out_embed_dim=embed_dim,
            rngs=rngs,
        )
        self.post_sa_norm = nnx.RMSNorm(
            num_features=embed_dim,
            epsilon=enc_config.norm_eps,
            dtype=jnp.float32,
            param_dtype=param_dtype,
            # scale_init=nnx.with_partitioning(nnx.initializers.ones_init(), ("model",)),
            rngs=rngs,
        )
        self.mlp = MlpBlock(
            embed_dim=embed_dim,
            intermediate_dim=enc_config.intermediate_size,
            compute_dtype=self.compute_dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def __call__(self, x: jnp.ndarray, state: EncoderInferenceState) -> jnp.ndarray:
        residual = x
        x_norm = self.pre_sa_norm(x).astype(self.compute_dtype)

        sa_out = self.self_attention(
            X=x_norm,
            q_positions=state.positions,
            kv_positions=state.positions,
            attn_mask=state.attn_mask,
        )

        x = residual + sa_out

        residual = x
        x_norm = self.post_sa_norm(x).astype(self.compute_dtype)
        mlp_out = self.mlp(x_norm)
        x = residual + mlp_out

        return x


class Encoder(nnx.Module):
    """Transformer Encoder Stack (nnx version) with scan optimization."""

    config: DiaConfig
    compute_dtype: jnp.dtype = jnp.float32

    def __init__(
        self,
        config: DiaConfig,
        compute_dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ):
        self.config = config
        enc_config = config.encoder_config
        self.compute_dtype = compute_dtype

        self.embedding = nnx.Embed(
            num_embeddings=enc_config.vocab_size,
            features=enc_config.hidden_size,
            dtype=self.compute_dtype,
            param_dtype=param_dtype,
            # embedding_init=nnx.with_partitioning(
            #     nnx.initializers.variance_scaling(1.0, "fan_in", "uniform"),
            #     ("model", None),
            # ),
            rngs=rngs,
        )

        # Create encoder layers
        self.layers = [
            EncoderLayer(
                config=self.config,
                compute_dtype=self.compute_dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            )
            for _ in range(enc_config.num_hidden_layers)
        ]

        self.norm = nnx.RMSNorm(
            num_features=enc_config.hidden_size,
            epsilon=enc_config.norm_eps,
            dtype=jnp.float32,
            param_dtype=param_dtype,
            scale_init=nnx.with_partitioning(nnx.initializers.ones_init(), ("model",)),
            rngs=rngs,
        )

    def __call__(self, x_ids: jnp.ndarray, state: EncoderInferenceState) -> jnp.ndarray:
        x = self.embedding(x_ids)

        # Process through all encoder layers
        for layer in self.layers:
            x = layer(x, state)

        x = self.norm(x).astype(self.compute_dtype)
        return x


class DecoderLayer(nnx.Module):
    """Transformer Decoder Layer."""

    config: DiaConfig
    compute_dtype: jnp.dtype = jnp.float32

    def __init__(
        self,
        config: DiaConfig,
        compute_dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ):
        self.config = config
        dec_config = self.config.decoder_config
        enc_config = self.config.encoder_config
        dec_embed_dim = dec_config.hidden_size
        enc_embed_dim = enc_config.hidden_size
        self.compute_dtype = compute_dtype

        # Norms
        self.pre_sa_norm = nnx.RMSNorm(
            num_features=dec_embed_dim,
            epsilon=dec_config.norm_eps,
            dtype=jnp.float32,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.pre_ca_norm = nnx.RMSNorm(
            num_features=dec_embed_dim,
            epsilon=dec_config.norm_eps,
            dtype=jnp.float32,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.pre_mlp_norm = nnx.RMSNorm(
            num_features=dec_embed_dim,
            epsilon=dec_config.norm_eps,
            dtype=jnp.float32,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        # Self-Attention (GQA) with Causal Masking
        self.self_attention = SelfAttention(
            config=self.config.decoder_config,
            q_embed_dim=dec_embed_dim,
            kv_embed_dim=dec_embed_dim,
            num_query_heads=dec_config.num_attention_heads,
            num_kv_heads=dec_config.num_key_value_heads,
            head_dim=dec_config.head_dim,
            compute_dtype=self.compute_dtype,
            param_dtype=param_dtype,
            out_embed_dim=dec_embed_dim,
            rngs=rngs,
        )
        # Cross-Attention (MHA)
        self.cross_attention = CrossAttention(
            config=self.config.decoder_config,
            q_embed_dim=dec_embed_dim,
            kv_embed_dim=enc_embed_dim,
            num_query_heads=dec_config.cross_num_attention_heads,
            num_kv_heads=dec_config.cross_num_key_value_heads,
            head_dim=dec_config.cross_head_dim,
            compute_dtype=self.compute_dtype,
            param_dtype=param_dtype,
            out_embed_dim=dec_embed_dim,
            rngs=rngs,
        )
        self.mlp = MlpBlock(
            embed_dim=dec_embed_dim,
            intermediate_dim=dec_config.intermediate_size,
            compute_dtype=self.compute_dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def __call__(
        self,
        x: jnp.ndarray,
        state: DecoderInferenceState,
        self_attn_cache: KVCache | None = None,
        cross_attn_cache: KVCache | None = None,
        prefill: bool = False,
        current_idx: int = 0,
    ) -> jnp.ndarray:
        residual = x
        x_norm = self.pre_sa_norm(x).astype(self.compute_dtype)
        if prefill:
            self_attn_mask = state.causal_attn_mask
        else:
            self_attn_mask = state.causal_attn_mask[None, None, current_idx]
        sa_out = self.self_attention(
            X=x_norm,
            q_positions=state.dec_positions,
            kv_positions=state.dec_positions,
            attn_mask=self_attn_mask,
            cache=self_attn_cache,
            prefill=prefill,
            is_causal=prefill,
            current_idx=current_idx,
        )
        x = residual + sa_out
        residual = x
        x_norm = self.pre_ca_norm(x).astype(self.compute_dtype)
        ca_out = self.cross_attention(
            Xq=x_norm,
            q_positions=state.dec_positions,
            kv_positions=state.enc_positions,
            attn_mask=state.cross_attn_mask,
            cache=cross_attn_cache,
            encoder_output=state.enc_out,
        )
        x = residual + ca_out
        residual = x
        x_norm = self.pre_mlp_norm(x).astype(self.compute_dtype)
        mlp_out = self.mlp(x_norm)
        x = residual + mlp_out

        return x

    def forward_jit(
        self,
        x: jnp.ndarray,
        current_idx: int,
        causal_attn_mask: jnp.ndarray,
        cross_attn_mask: jnp.ndarray,
        dec_positions: jnp.ndarray,
        enc_positions: jnp.ndarray,
        enc_out: jnp.ndarray,
        self_attn_cache: KVCache | None = None,
        cross_attn_cache: KVCache | None = None,
    ) -> jnp.ndarray:
        residual = x
        x_norm = self.pre_sa_norm(x).astype(self.compute_dtype)
        self_attn_mask = causal_attn_mask[None, None, current_idx]
        sa_out = self.self_attention(
            X=x_norm,
            q_positions=dec_positions,
            kv_positions=dec_positions,
            attn_mask=self_attn_mask,
            cache=self_attn_cache,
            prefill=False,
            is_causal=False,
            current_idx=current_idx,
        )
        x = residual + sa_out
        residual = x
        x_norm = self.pre_ca_norm(x).astype(self.compute_dtype)
        ca_out = self.cross_attention(
            Xq=x_norm,
            q_positions=dec_positions,
            kv_positions=enc_positions,
            attn_mask=cross_attn_mask,
            cache=cross_attn_cache,
            encoder_output=enc_out,
        )
        x = residual + ca_out
        residual = x
        x_norm = self.pre_mlp_norm(x).astype(self.compute_dtype)
        mlp_out = self.mlp(x_norm)
        x = residual + mlp_out

        return x


class Decoder(nnx.Module):
    """Transformer Decoder Stack."""

    config: DiaConfig
    compute_dtype: jnp.dtype = jnp.float32

    def __init__(
        self,
        config: DiaConfig,
        compute_dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ):
        self.config = config
        self.compute_dtype = compute_dtype
        dec_config = self.config.decoder_config
        self.num_channels = dec_config.num_channels
        self.num_layers = dec_config.num_hidden_layers

        self.embeddings = [
            nnx.Embed(
                num_embeddings=dec_config.vocab_size,
                features=dec_config.hidden_size,
                dtype=self.compute_dtype,
                param_dtype=param_dtype,
                # embedding_init=nnx.with_partitioning(
                #     nnx.initializers.variance_scaling(1.0, "fan_in", "uniform"),
                #     ("model", None),
                # ),
                rngs=rngs,
            )
            for _ in range(self.num_channels)
        ]
        self.layers = [
            DecoderLayer(
                config=self.config,
                compute_dtype=self.compute_dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            )
            for _ in range(self.num_layers)
        ]
        self.norm = nnx.RMSNorm(
            num_features=dec_config.hidden_size,
            epsilon=dec_config.norm_eps,
            dtype=jnp.float32,
            param_dtype=param_dtype,
            # scale_init=nnx.with_partitioning(nnx.initializers.ones_init(), ("model",)),
            rngs=rngs,
        )
        ki = nnx.initializers.variance_scaling(1.0, "fan_in", "uniform")
        self.logits_dense = nnx.LinearGeneral(
            in_features=(dec_config.hidden_size,),
            out_features=(self.num_channels, dec_config.vocab_size),
            axis=(-1,),
            use_bias=False,
            param_dtype=param_dtype,
            dtype=compute_dtype,
            # kernel_init=nnx.with_partitioning(ki, (None, None, "model")),
            rngs=rngs,
        )

    def precompute_cross_attn_cache(
        self,
        enc_out: jnp.ndarray,  # (B, S, E)
    ) -> list[KVCache]:
        """
        Computes the Key and Value tensors for cross-attention for each layer from the encoder output.
        """
        per_layer_kv_cache: list[KVCache] = []

        for layer in self.layers:
            cross_attn_module = layer.cross_attention
            k = cross_attn_module.k_proj(enc_out)
            v = cross_attn_module.v_proj(enc_out)
            per_layer_kv_cache.append(KVCache.from_kv(k, v))

        return per_layer_kv_cache

    def decode_step(
        self,
        tgt_ids_Bx1xC: jnp.ndarray,  # (B, 1, C)
        state: DecoderInferenceState,
        current_idx: int,
    ) -> jnp.ndarray:
        """
        Performs a single decoding step, managing KV caches layer by layer.

        Returns:
            logits_Bx1xCxV: The final output logits for the current step (B, 1, C, V).
        """
        x = None
        for i in range(self.num_channels):
            channel_tokens = tgt_ids_Bx1xC[..., i]
            channel_embed = self.embeddings[i](channel_tokens)
            x = channel_embed if x is None else x + channel_embed

        for i, layer in enumerate(self.layers):
            self_cache = state.self_attn_cache[i]
            cross_cache = state.cross_attn_cache[i]
            x = layer(
                x,
                state,
                self_attn_cache=self_cache,
                cross_attn_cache=cross_cache,
                current_idx=current_idx,
            )

        x = self.norm(x).astype(self.compute_dtype)
        logits_Bx1xCxV = self.logits_dense(x)
        return logits_Bx1xCxV

    def decode_step_jit(
        self,
        tgt_ids_Bx1xC: jnp.ndarray,
        current_idx: int,
        causal_attn_mask: jnp.ndarray,
        cross_attn_mask: jnp.ndarray,
        dec_positions: jnp.ndarray,
        enc_positions: jnp.ndarray,
        enc_out: jnp.ndarray,
        self_attn_cache: list[KVCache],
        cross_attn_cache: list[KVCache],
    ):
        x = None
        for i in range(self.num_channels):
            channel_tokens = tgt_ids_Bx1xC[..., i]
            channel_embed = self.embeddings[i](channel_tokens)
            x = channel_embed if x is None else x + channel_embed

        for i, layer in enumerate(self.layers):
            self_cache = self_attn_cache[i]
            cross_cache = cross_attn_cache[i]
            x = layer.forward_jit(
                x,
                current_idx=current_idx,
                causal_attn_mask=causal_attn_mask,
                cross_attn_mask=cross_attn_mask,
                dec_positions=dec_positions,
                enc_positions=enc_positions,
                enc_out=enc_out,
                self_attn_cache=self_cache,
                cross_attn_cache=cross_cache,
            )

        x = self.norm(x).astype(self.compute_dtype)
        logits_Bx1xCxV = self.logits_dense(x)
        return logits_Bx1xCxV, self_attn_cache, cross_attn_cache

    def __call__(self, tgt_ids_BxTxC: jnp.ndarray, state) -> jnp.ndarray:
        """
        Forward pass for the Decoder stack.

        Args:
            tgt_ids_BxTxC: Target token IDs (B, T, C).
            state: DecoderInferenceState or DecoderTrainingState containing state information.

        Returns:
            logits: The final output logits (B, T, C, V).
        """
        _, _, num_channels_in = tgt_ids_BxTxC.shape
        assert num_channels_in == self.num_channels, "Input channels mismatch"

        # Embeddings
        x = None
        for i in range(self.num_channels):
            channel_tokens = tgt_ids_BxTxC[..., i]
            channel_embed = self.embeddings[i](channel_tokens)
            x = channel_embed if x is None else x + channel_embed

        # In training mode the cache is None
        for i, layer in enumerate(self.layers):
            self_cache = state.self_attn_cache[i]
            cross_cache = state.cross_attn_cache[i]

            x = layer(
                x,
                state,
                self_attn_cache=self_cache,
                cross_attn_cache=cross_cache,
                prefill=True,
            )

        # Final Norm
        x = self.norm(x).astype(self.compute_dtype)
        logits_BxTxCxV = self.logits_dense(x)

        return logits_BxTxCxV


class DiaModel(nnx.Module):
    """JAX/Flax Dia Model"""

    config: DiaConfig
    compute_dtype: jnp.dtype = jnp.float32

    def __init__(
        self,
        config: DiaConfig,
        compute_dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ):
        self.config = config
        self.compute_dtype = compute_dtype
        self.param_dtype = param_dtype

        self.encoder = Encoder(
            config=self.config,
            compute_dtype=self.compute_dtype,
            param_dtype=self.param_dtype,
            rngs=rngs,
        )
        self.decoder = Decoder(
            config=self.config,
            compute_dtype=self.compute_dtype,
            param_dtype=self.param_dtype,
            rngs=rngs,
        )
