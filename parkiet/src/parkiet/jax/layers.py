import jax
import jax.numpy as jnp
from flax import linen as nn
from parkiet.dia.config import DiaConfig
from parkiet.jax.state import DecoderInferenceState, EncoderInferenceState
from jax.nn import dot_product_attention


class MlpBlock(nn.Module):
    """MLP block using DenseGeneral."""

    embed_dim: int
    intermediate_dim: int
    compute_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass."""
        # Fused gate and up projection
        wi_fused = nn.DenseGeneral(
            features=(2, self.intermediate_dim),
            axis=(-1,),
            dtype=self.compute_dtype,
            name="wi_fused",
            use_bias=False,
        )(x)

        gate = wi_fused[..., 0, :]
        up = wi_fused[..., 1, :]

        hidden = jax.nn.silu(gate) * up

        # Output projection
        output = nn.DenseGeneral(
            features=(self.embed_dim,),
            axis=(-1,),
            dtype=self.compute_dtype,
            name="wo",
            use_bias=False,
        )(hidden)

        return output


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) implementation in JAX."""

    embedding_dims: int
    min_timescale: int = 1
    max_timescale: int = 10000
    compute_dtype: jnp.dtype = jnp.float32

    def setup(self):
        if self.embedding_dims % 2 != 0:
            raise ValueError("Embedding dim must be even for RoPE.")

        half_embedding_dim = self.embedding_dims // 2
        fraction = (2.0 * jnp.arange(0, half_embedding_dim)) / self.embedding_dims
        timescale = (
            self.min_timescale * (self.max_timescale / self.min_timescale) ** fraction
        ).astype(jnp.float32)
        self.timescale = self.variable(
            "constants", "timescale", lambda: timescale.astype(jnp.float32)
        )

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


class CrossAttention(nn.Module):
    """Cross-Attention using DenseGeneral."""

    config: DiaConfig
    q_embed_dim: int
    kv_embed_dim: int
    num_query_heads: int
    num_kv_heads: int
    head_dim: int
    compute_dtype: jnp.dtype = jnp.float32
    out_embed_dim: int | None = None

    def setup(self):
        self.output_dim = (
            self.out_embed_dim if self.out_embed_dim is not None else self.q_embed_dim
        )
        self.projected_query_dim = self.num_query_heads * self.head_dim
        if self.num_query_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_query_heads ({self.num_query_heads}) must be divisible by num_kv_heads ({self.num_kv_heads})"
            )
        self.num_gqa_groups = self.num_query_heads // self.num_kv_heads

        # Projection layers
        self.q_proj = nn.DenseGeneral(
            features=(self.num_query_heads, self.head_dim),
            axis=(-1,),
            dtype=self.compute_dtype,
            name="q_proj",
        )
        self.k_proj = nn.DenseGeneral(
            features=(self.num_kv_heads, self.head_dim),
            axis=(-1,),
            dtype=self.compute_dtype,
            name="k_proj",
        )
        self.v_proj = nn.DenseGeneral(
            features=(self.num_kv_heads, self.head_dim),
            axis=(-1,),
            dtype=self.compute_dtype,
            name="v_proj",
        )
        self.o_proj = nn.DenseGeneral(
            features=(self.output_dim,),
            axis=(-2, -1),
            dtype=self.compute_dtype,
            name="o_proj",
        )

        # Rotary embedding
        self.rotary_emb = RotaryEmbedding(
            embedding_dims=self.head_dim,
            min_timescale=self.config.model.rope_min_timescale,
            max_timescale=self.config.model.rope_max_timescale,
            compute_dtype=self.compute_dtype,
        )

    def __call__(
        self,
        Xq: jnp.ndarray,  # (B, T, D) T = 1 in AR generation
        q_positions: jnp.ndarray,  # (B, T)
        kv_positions: jnp.ndarray | None = None,  # (B, S)
        attn_mask: jnp.ndarray
        | None = None,  # None in Decoder Self Attention, Valid mask in Others
        cache_k: jnp.ndarray | None = None,  # Cached keys
        cache_v: jnp.ndarray | None = None,  # Cached values
        is_causal: bool = False,
    ) -> jnp.ndarray:
        if kv_positions is None:
            kv_positions = q_positions
        original_dtype = Xq.dtype

        Xq_BxTxNxH = self.q_proj(Xq)
        Xq_BxTxNxH = self.rotary_emb(Xq_BxTxNxH, position=q_positions)
        Xq_BxNxTxH = jnp.swapaxes(Xq_BxTxNxH, 1, 2)

        attn_k = cache_k
        attn_v = cache_v

        attn_output = dot_product_attention(
            query=Xq_BxNxTxH,
            key=attn_k,
            value=attn_v,
            mask=attn_mask if is_causal else None,
            scale=1.0,
            is_causal=is_causal,
        )

        attn_output = jnp.swapaxes(attn_output, 1, 2)  # (B, T, N, H)
        output = self.o_proj(attn_output)

        return output.astype(original_dtype)


class SelfAttention(nn.Module):
    """Self-Attention using DenseGeneral."""

    config: DiaConfig
    q_embed_dim: int
    kv_embed_dim: int
    num_query_heads: int
    num_kv_heads: int
    head_dim: int
    compute_dtype: jnp.dtype = jnp.float32
    is_cross_attn: bool = False
    out_embed_dim: int | None = None

    def setup(self):
        self.output_dim = (
            self.out_embed_dim if self.out_embed_dim is not None else self.q_embed_dim
        )
        self.projected_query_dim = self.num_query_heads * self.head_dim
        if self.num_query_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_query_heads ({self.num_query_heads}) must be divisible by num_kv_heads ({self.num_kv_heads})"
            )
        self.num_gqa_groups = self.num_query_heads // self.num_kv_heads

        # Projection layers
        self.q_proj = nn.DenseGeneral(
            features=(self.num_query_heads, self.head_dim),
            axis=(-1,),
            dtype=self.compute_dtype,
            name="q_proj",
            use_bias=False,
        )
        self.k_proj = nn.DenseGeneral(
            features=(self.num_kv_heads, self.head_dim),
            axis=(-1,),
            dtype=self.compute_dtype,
            name="k_proj",
            use_bias=False,
        )
        self.v_proj = nn.DenseGeneral(
            features=(self.num_kv_heads, self.head_dim),
            axis=(-1,),
            dtype=self.compute_dtype,
            name="v_proj",
            use_bias=False,
        )
        self.o_proj = nn.DenseGeneral(
            features=(self.output_dim,),
            axis=(-2, -1),
            dtype=self.compute_dtype,
            name="o_proj",
            use_bias=False,
        )

        self.rotary_emb = RotaryEmbedding(
            embedding_dims=self.head_dim,
            min_timescale=self.config.model.rope_min_timescale,
            max_timescale=self.config.model.rope_max_timescale,
            compute_dtype=self.compute_dtype,
        )

    def __call__(
        self,
        X: jnp.ndarray,  # (B, T, D) T = 1 in AR generation
        q_positions: jnp.ndarray,  # (B, T)
        kv_positions: jnp.ndarray | None = None,  # (B, S)
        attn_mask: jnp.ndarray
        | None = None,  # None in Decoder Self Attention, Valid mask in Others
        cache_k: jnp.ndarray | None = None,  # Cached keys
        cache_v: jnp.ndarray | None = None,  # Cached values
        prefill: bool = False,
        is_causal: bool = False,
        current_idx: int | None = None,
    ) -> jnp.ndarray:
        if kv_positions is None:
            kv_positions = q_positions

        original_dtype = X.dtype

        Xq_BxTxNxH = self.q_proj(X)
        Xk_BxSxKxH = self.k_proj(X)
        Xv_BxSxKxH = self.v_proj(X)

        # Apply RoPE
        position = q_positions[..., None, None]
        sinusoid_inp = position / self.rotary_emb.timescale.value
        sin = jnp.sin(sinusoid_inp)
        cos = jnp.cos(sinusoid_inp)

        Xq_BxTxNxH = self.rotary_emb.apply_rope(Xq_BxTxNxH, sin, cos)
        Xk_BxSxKxH = self.rotary_emb.apply_rope(Xk_BxSxKxH, sin, cos)

        if cache_k is None:
            attn_k = Xk_BxSxKxH
            attn_v = Xv_BxSxKxH
        else:
            # Use cached values (for inference)
            attn_k = cache_k
            attn_v = cache_v

        attn_output = dot_product_attention(
            query=Xq_BxTxNxH,
            key=attn_k,
            value=attn_v,
            mask=attn_mask if is_causal else None,
            scale=1.0,
            is_causal=is_causal,
        )

        output = self.o_proj(attn_output)
        return output.astype(original_dtype)


class EncoderLayer(nn.Module):
    """Transformer Encoder Layer."""

    config: DiaConfig
    compute_dtype: jnp.dtype = jnp.float32

    def setup(self):
        model_config = self.config.model
        enc_config = self.config.model.encoder
        embed_dim = enc_config.n_embd

        self.pre_sa_norm = nn.RMSNorm(
            epsilon=model_config.normalization_layer_epsilon,
            dtype=jnp.float32,
            name="pre_sa_norm",
        )
        self.self_attention = SelfAttention(
            config=self.config,
            q_embed_dim=embed_dim,
            kv_embed_dim=embed_dim,
            num_query_heads=enc_config.n_head,
            num_kv_heads=enc_config.n_head,
            head_dim=enc_config.head_dim,
            compute_dtype=self.compute_dtype,
            is_cross_attn=False,
            out_embed_dim=embed_dim,
        )
        self.post_sa_norm = nn.RMSNorm(
            epsilon=model_config.normalization_layer_epsilon,
            dtype=jnp.float32,
            name="post_sa_norm",
        )
        self.mlp = MlpBlock(
            embed_dim=embed_dim,
            intermediate_dim=enc_config.n_hidden,
            compute_dtype=self.compute_dtype,
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


class Encoder(nn.Module):
    """Transformer Encoder Stack."""

    config: DiaConfig
    compute_dtype: jnp.dtype = jnp.float32

    def setup(self):
        model_config = self.config.model
        enc_config = self.config.model.encoder

        self.embedding = nn.Embed(
            num_embeddings=model_config.src_vocab_size,
            features=enc_config.n_embd,
            dtype=self.compute_dtype,
            name="embedding",
        )
        self.layers = [
            EncoderLayer(
                config=self.config, compute_dtype=self.compute_dtype, name=f"layers.{i}"
            )
            for i in range(enc_config.n_layer)
        ]
        self.norm = nn.RMSNorm(
            epsilon=model_config.normalization_layer_epsilon,
            dtype=jnp.float32,
            name="norm",
        )

    def __call__(self, x_ids: jnp.ndarray, state: EncoderInferenceState) -> jnp.ndarray:
        x = self.embedding(x_ids)

        for layer in self.layers:
            x = layer(x, state)

        x = self.norm(x).astype(self.compute_dtype)
        return x


class DecoderLayer(nn.Module):
    """Transformer Decoder Layer."""

    config: DiaConfig
    compute_dtype: jnp.dtype = jnp.float32

    def setup(self):
        model_config = self.config.model
        dec_config = self.config.model.decoder
        enc_config = self.config.model.encoder
        dec_embed_dim = dec_config.n_embd
        enc_embed_dim = enc_config.n_embd

        # Norms
        self.pre_sa_norm = nn.RMSNorm(
            epsilon=model_config.normalization_layer_epsilon,
            dtype=jnp.float32,
            name="pre_sa_norm",
        )
        self.pre_ca_norm = nn.RMSNorm(
            epsilon=model_config.normalization_layer_epsilon,
            dtype=jnp.float32,
            name="pre_ca_norm",
        )
        self.pre_mlp_norm = nn.RMSNorm(
            epsilon=model_config.normalization_layer_epsilon,
            dtype=jnp.float32,
            name="pre_mlp_norm",
        )

        # Self-Attention (GQA) with Causal Masking
        self.self_attention = SelfAttention(
            config=self.config,
            q_embed_dim=dec_embed_dim,
            kv_embed_dim=dec_embed_dim,
            num_query_heads=dec_config.gqa_query_heads,
            num_kv_heads=dec_config.kv_heads,
            head_dim=dec_config.gqa_head_dim,
            compute_dtype=self.compute_dtype,
            is_cross_attn=False,
            out_embed_dim=dec_embed_dim,
        )
        # Cross-Attention (MHA)
        self.cross_attention = CrossAttention(
            config=self.config,
            q_embed_dim=dec_embed_dim,
            kv_embed_dim=enc_embed_dim,
            num_query_heads=dec_config.cross_query_heads,
            num_kv_heads=dec_config.cross_query_heads,
            head_dim=dec_config.cross_head_dim,
            compute_dtype=self.compute_dtype,
            out_embed_dim=dec_embed_dim,
        )
        # MLP
        self.mlp = MlpBlock(
            embed_dim=dec_embed_dim,
            intermediate_dim=dec_config.n_hidden,
            compute_dtype=self.compute_dtype,
        )

    def __call__(
        self,
        x: jnp.ndarray,
        state: DecoderInferenceState,
        self_attn_cache_k: jnp.ndarray | None = None,
        self_attn_cache_v: jnp.ndarray | None = None,
        cross_attn_cache_k: jnp.ndarray | None = None,
        cross_attn_cache_v: jnp.ndarray | None = None,
        prefill: bool = False,
        current_idx: int = 0,
    ) -> jnp.ndarray:
        residual = x
        x_norm = self.pre_sa_norm(x).astype(self.compute_dtype)

        # Self attention with causal mask
        if prefill:
            self_attn_mask = None  # Use is_causal=True instead
        else:
            self_attn_mask = state.casual_attn_mask[None, None, current_idx]

        sa_out = self.self_attention(
            X=x_norm,
            q_positions=state.dec_positions,
            kv_positions=state.dec_positions,
            attn_mask=self_attn_mask,
            cache_k=self_attn_cache_k,
            cache_v=self_attn_cache_v,
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
            cache_k=cross_attn_cache_k,
            cache_v=cross_attn_cache_v,
        )
        x = residual + ca_out

        residual = x
        x_norm = self.pre_mlp_norm(x).astype(self.compute_dtype)
        mlp_out = self.mlp(x_norm)
        x = residual + mlp_out

        return x


class Decoder(nn.Module):
    """Transformer Decoder Stack."""

    config: DiaConfig
    compute_dtype: jnp.dtype = jnp.float32

    def setup(self):
        model_config = self.config.model
        dec_config = self.config.model.decoder
        data_config = self.config.data
        self.num_channels = data_config.channels
        self.num_layers = dec_config.n_layer

        self.embeddings = [
            nn.Embed(
                num_embeddings=model_config.tgt_vocab_size,
                features=dec_config.n_embd,
                dtype=self.compute_dtype,
                name=f"embedding_{i}",
            )
            for i in range(self.num_channels)
        ]
        self.layers = [
            DecoderLayer(
                config=self.config, compute_dtype=self.compute_dtype, name=f"layer_{i}"
            )
            for i in range(self.num_layers)
        ]

        self.norm = nn.RMSNorm(
            epsilon=model_config.normalization_layer_epsilon,
            dtype=jnp.float32,
            name="norm",
        )

        self.logits_dense = nn.DenseGeneral(
            features=(self.num_channels, model_config.tgt_vocab_size),
            axis=(-1,),
            dtype=self.compute_dtype,
            name="logits_dense",
        )

    def __call__(
        self,
        tgt_ids_BxTxC: jnp.ndarray,
        state: DecoderInferenceState,
        cross_attn_caches: list | None = None,
        self_attn_caches: list | None = None,
        prefill: bool = False,
    ) -> jnp.ndarray:
        """
        Forward pass for the Decoder stack.

        Args:
            tgt_ids_BxTxC: Target token IDs (B, T, C).
            state: DecoderInferenceState containing all state information.
            cross_attn_caches: Precomputed cross-attention caches.
            self_attn_caches: Self-attention caches for incremental decoding.
            prefill: Whether this is prefill mode.

        Returns:
            logits: The final output logits (B, T, C * V), cast to float32.
        """
        _, _, num_channels_in = tgt_ids_BxTxC.shape
        assert num_channels_in == self.num_channels, "Input channels mismatch"

        # Embeddings
        x = None
        for i in range(self.num_channels):
            channel_tokens = tgt_ids_BxTxC[..., i]
            channel_embed = self.embeddings[i](channel_tokens)
            x = channel_embed if x is None else x + channel_embed

        for i, layer in enumerate(self.layers):
            cross_cache_k = cross_attn_caches[i] if cross_attn_caches else None
            cross_cache_v = cross_attn_caches[i] if cross_attn_caches else None
            self_cache_k = self_attn_caches[i] if self_attn_caches else None
            self_cache_v = self_attn_caches[i] if self_attn_caches else None

            x = layer(
                x,
                state,
                self_attn_cache_k=self_cache_k,
                self_attn_cache_v=self_cache_v,
                cross_attn_cache_k=cross_cache_k,
                cross_attn_cache_v=cross_cache_v,
                prefill=prefill,
            )

        # Final Norm
        x = self.norm(x)
        logits_BxTxCxV = self.logits_dense(x)

        return logits_BxTxCxV.astype(jnp.float32)


class DiaModel(nn.Module):
    """JAX/Flax Dia Model"""

    config: DiaConfig
    compute_dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.encoder = Encoder(config=self.config, compute_dtype=self.compute_dtype)
        self.decoder = Decoder(config=self.config, compute_dtype=self.compute_dtype)
