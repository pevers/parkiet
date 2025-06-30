import os
import torch
import numpy as np
import jax
import jax.numpy as jnp
from parkiet.dia.config import DiaConfig
from parkiet.dia.model import Dia
import flax.nnx as nnx
import parkiet.jax.state as jax_state
import parkiet.jax.layers as jax_layers
from parkiet.jax.model import convert_torch_to_nnx
from jax.nn import dot_product_attention
import parkiet.dia.layers as dia_layers
import parkiet.dia.state as dia_state

# For debugging
np.set_printoptions(precision=10, suppress=False)
torch.set_printoptions(
    precision=15,  # number of digits after the decimal point
    sci_mode=False,  # turn off scientific notation
)
jax.config.update("jax_enable_x64", True)


def test_mlp_block_equivalence():
    dia_config = DiaConfig.load(
        os.path.join(os.path.dirname(__file__), "../config.json")
    )
    x = np.ones((1, 128), dtype=np.float32)
    torch_mlp_block = dia_layers.MlpBlock(
        embed_dim=128, intermediate_dim=256, compute_dtype=torch.float32
    )
    torch_mlp_block.eval()
    torch.nn.init.xavier_uniform(torch_mlp_block.wi_fused.weight)
    torch.nn.init.xavier_uniform(torch_mlp_block.wo.weight)
    torch_out = torch_mlp_block(torch.as_tensor(x)).detach().numpy()
    rngs = nnx.Rngs(0)
    jax_mlp_block = jax_layers.MlpBlock(
        embed_dim=128, intermediate_dim=256, compute_dtype=jnp.float32, rngs=rngs
    )
    jax_mlp_block = convert_torch_to_nnx(
        torch_mlp_block, jax_mlp_block, dia_config=dia_config
    )
    jax_mlp_block.eval()
    jax_out = jax_mlp_block(jnp.array(x))
    np.testing.assert_allclose(jax_out, torch_out, rtol=1e-5, atol=1e-5)


def test_rotary_embedding_equivalence():
    dia_config = DiaConfig.load(
        os.path.join(os.path.dirname(__file__), "../config.json")
    )
    x = np.ones((1, 128), dtype=np.float64)
    positions = np.random.randint(1, 10_000, (1, 128), dtype=np.int32)
    positions = positions.astype(np.float64)
    torch_rotary_embedding = dia_layers.RotaryEmbedding(
        embedding_dims=128, min_timescale=1, max_timescale=10_000, dtype=torch.float64
    )
    torch_out = torch_rotary_embedding(
        torch.as_tensor(x), torch.as_tensor(positions)
    ).numpy()
    rngs = nnx.Rngs(0)
    jax_rotary_embedding = jax_layers.RotaryEmbedding(
        embedding_dims=128,
        min_timescale=1,
        max_timescale=10_000,
        compute_dtype=jnp.float64,
        rngs=rngs,
    )
    jax_rotary_embedding = convert_torch_to_nnx(
        torch_rotary_embedding, jax_rotary_embedding, dia_config=dia_config
    )
    jax_out = jax_rotary_embedding(jnp.array(x), jnp.array(positions))
    np.testing.assert_allclose(jax_out, torch_out, rtol=1e-4, atol=1e-4)


def test_dot_product_attention_equivalence():
    # PyTorch expects: [batch, num_heads, seq_len, head_dim]
    # JAX expects: [batch, seq_len, num_heads, head_dim]
    query = np.random.uniform(-1, 1, (1, 1, 10, 2)).astype(
        np.float64
    )  # [batch, heads, seq_len, head_dim]
    key = np.random.uniform(-1, 1, (1, 1, 10, 2)).astype(np.float64)
    value = np.random.uniform(-1, 1, (1, 1, 10, 2)).astype(np.float64)
    torch_out = (
        torch.nn.functional.scaled_dot_product_attention(
            torch.as_tensor(query),
            torch.as_tensor(key),
            torch.as_tensor(value),
            attn_mask=None,
            scale=1.0,
            enable_gqa=False,
            is_causal=False,
        )
        .detach()
        .numpy()
    )

    # Convert from PyTorch format [batch, heads, seq_len, head_dim]
    # to JAX format [batch, seq_len, heads, head_dim]
    query_jax = query.transpose(0, 2, 1, 3)  # [batch, seq_len, heads, head_dim]
    key_jax = key.transpose(0, 2, 1, 3)
    value_jax = value.transpose(0, 2, 1, 3)

    jax_out = dot_product_attention(
        jnp.array(query_jax),
        jnp.array(key_jax),
        jnp.array(value_jax),
        bias=None,
        mask=None,
        scale=1.0,
        is_causal=False,
    )

    # Convert JAX output back to PyTorch format for comparison
    jax_out_torch_format = jax_out.transpose(
        0, 2, 1, 3
    )  # [batch, heads, seq_len, head_dim]
    np.testing.assert_allclose(jax_out_torch_format, torch_out, rtol=1e-4, atol=1e-4)


def test_rms_equivalence():
    x = np.random.uniform(-1, 1, (2, 1024, 1024)).astype(np.float32)
    torch_rms = torch.nn.RMSNorm(normalized_shape=1024, eps=1e-5, dtype=torch.float32)
    torch_out = torch_rms(torch.as_tensor(x)).detach().numpy()
    rngs = nnx.Rngs(0)
    jax_rms = nnx.RMSNorm(num_features=1024, epsilon=1e-5, dtype=jnp.float32, rngs=rngs)
    jax_out = jax_rms(jnp.array(x))
    np.testing.assert_allclose(jax_out, torch_out, rtol=1e-4, atol=1e-4)


def test_cross_attention_equivalence():
    raise NotImplementedError("Cross attention is not implemented in JAX")


def test_self_attention_equivalence():
    dia_config = DiaConfig.load(
        os.path.join(os.path.dirname(__file__), "../config.json")
    )
    X = np.random.uniform(-1, 1, (2, 10, 1024)).astype(np.float32)
    q_positions = np.arange(10, dtype=np.float32)
    kv_positions = np.arange(10, dtype=np.float32)

    torch_self_attention = dia_layers.SelfAttention(
        config=dia_config.encoder_config,
        q_embed_dim=1024,
        kv_embed_dim=1024,
        num_query_heads=16,
        num_kv_heads=16,
        head_dim=64,
        compute_dtype=torch.float32,
    )
    torch.nn.init.xavier_uniform_(torch_self_attention.q_proj.weight)
    torch.nn.init.xavier_uniform_(torch_self_attention.k_proj.weight)
    torch.nn.init.xavier_uniform_(torch_self_attention.v_proj.weight)
    torch.nn.init.xavier_uniform_(torch_self_attention.o_proj.weight)
    torch_out = (
        torch_self_attention(
            torch.as_tensor(X),
            torch.as_tensor(q_positions),
            torch.as_tensor(kv_positions),
            None,  # Mask is None means a causal mask is used
            None,  # No cache
        )
        .detach()
        .numpy()
    )
    rngs = nnx.Rngs(0)
    jax_self_attention = jax_layers.SelfAttention(
        config=dia_config.encoder_config,
        q_embed_dim=1024,
        kv_embed_dim=1024,
        num_query_heads=16,
        num_kv_heads=16,
        head_dim=64,
        compute_dtype=jnp.float32,
        rngs=rngs,
    )
    jax_self_attention = convert_torch_to_nnx(
        torch_self_attention, jax_self_attention, dia_config=dia_config
    )
    jax_out = jax_self_attention(
        jnp.array(X),
        jnp.array(q_positions),
        jnp.array(kv_positions),
        None,
    )
    np.testing.assert_allclose(jax_out, torch_out, rtol=1e-4, atol=1e-4)


def test_encoder_layer_equivalence():
    # Load config and weights
    dia = Dia.from_local(
        config_path=os.path.join(os.path.dirname(__file__), "../config.json"),
        checkpoint_path=os.path.join(
            os.path.dirname(__file__), "../weights/dia-v1.pth"
        ),
        compute_dtype="float32",
        device="cpu",
        load_dac=False,
    )
    dia_config = dia.config
    vocab_size = dia_config.encoder_config.vocab_size
    seq_len = dia_config.encoder_config.max_position_embeddings

    input_tokens = torch.randint(0, vocab_size, (1, 1, seq_len), dtype=torch.long)
    torch_inference_state = dia_state.EncoderInferenceState.new(
        dia_config, input_tokens
    )

    # PyTorch
    torch_layer = dia_layers.EncoderLayer(dia_config, compute_dtype=torch.float32)
    torch_layer.eval()
    embedding = np.random.uniform(-1, 1, (2, 1024, 1024)).astype(np.float32)
    torch_out = (
        torch_layer(torch.as_tensor(embedding), torch_inference_state).detach().numpy()
    )

    # JAX
    rngs = nnx.Rngs(0)
    jax_layer = jax_layers.EncoderLayer(
        config=dia_config, compute_dtype=jnp.float64, rngs=rngs
    )
    jax_layer = convert_torch_to_nnx(torch_layer, jax_layer, dia_config=dia_config)
    jax_inference_state = jax_state.EncoderInferenceState.new(
        dia_config, jnp.array(input_tokens)
    )
    jax_out = jax_layer(jnp.array(embedding), jax_inference_state)
    np.testing.assert_allclose(jax_out, torch_out, rtol=1e-4, atol=1e-4)


def test_fused_qkv_equivalence():
    raise NotImplementedError("Fused QKV is not implemented in JAX")


def test_self_attention_cache():
    pass


def test_encoder_equivalence():
    # Load config and weights
    dia = Dia.from_local(
        config_path=os.path.join(os.path.dirname(__file__), "../config.json"),
        checkpoint_path=os.path.join(
            os.path.dirname(__file__), "../weights/dia-v1.pth"
        ),
        compute_dtype="float32",
        device="cpu",
        load_dac=False,
    )

    # Instantiate PyTorch model and load weights
    dia_config = dia.config
    torch_model = dia.model
    torch_model.load_state_dict(dia.model.state_dict())
    torch_encoder = torch_model.encoder
    torch_encoder.eval()

    # Prepare input and state
    seq_len = dia_config.encoder_config.max_position_embeddings
    input_tokens = torch.ones((1, 1, seq_len), dtype=torch.long)
    torch_inference_state = dia_state.EncoderInferenceState.new(
        dia_config, input_tokens
    )

    with torch.no_grad():
        encoder_input = input_tokens.squeeze(1).repeat_interleave(2, dim=0)
        torch_out = torch_encoder(
            torch.as_tensor(encoder_input), torch_inference_state
        ).numpy()

    # Prepare Flax encoder and params
    rngs = nnx.Rngs(0)
    flax_encoder = jax_layers.Encoder(
        config=dia_config, compute_dtype=jnp.float64, rngs=rngs
    )
    jax_input_tokens = jnp.array(input_tokens.numpy())
    jax_inference_state = jax_state.EncoderInferenceState.new(
        dia_config, jax_input_tokens
    )
    jax_encoder_input = jax_input_tokens.squeeze(1).repeat(2, axis=0)  # For CFG

    jax_encoder = convert_torch_to_nnx(
        torch_encoder, flax_encoder, dia_config=dia_config
    )
    jax_out = jax_encoder(jax_encoder_input, jax_inference_state)

    # Compare outputs
    np.testing.assert_allclose(torch_out, jax_out, rtol=1e-3, atol=1e-3)


def test_decoder_layer_equivalence():
    # Load config and weights
    dia = Dia.from_local(
        config_path=os.path.join(os.path.dirname(__file__), "../config.json"),
        checkpoint_path=os.path.join(
            os.path.dirname(__file__), "../weights/dia-v1.pth"
        ),
        compute_dtype="float32",
        device="cpu",
        load_dac=False,
    )
    dia_config = dia.config
    dec_config = dia_config.decoder_config
    enc_config = dia_config.encoder_config
    batch_size = 2
    seq_len = 4
    dec_embed_dim = dec_config.hidden_size
    enc_embed_dim = enc_config.hidden_size

    # Random input and dummy encoder output
    enc_out = np.random.uniform(-1, 1, (batch_size * 2, seq_len, enc_embed_dim)).astype(
        np.float32
    )
    torch_inference_state = dia_state.EncoderInferenceState.new(
        dia_config, torch.ones((batch_size, 1, seq_len), dtype=torch.long)
    )

    # PyTorch state
    torch_layer = dia_layers.DecoderLayer(dia_config, compute_dtype=torch.float32)
    torch_layer.eval()

    # Create decoder and precompute cross-attn cache
    torch_decoder = dia_layers.Decoder(dia_config, compute_dtype=torch.float32)
    dec_cross_attn_cache_torch = torch_decoder.precompute_cross_attn_cache(
        torch.as_tensor(enc_out)
    )
    torch_state = dia_state.DecoderInferenceState.new(
        dia_config,
        enc_state=torch_inference_state,
        enc_out=torch.as_tensor(enc_out),
        dec_cross_attn_cache=dec_cross_attn_cache_torch,
        compute_dtype=torch.float32,
        max_generation_length=seq_len,
    )
    with torch.no_grad():
        x = np.random.uniform(-1, 1, (batch_size * 2, seq_len, dec_embed_dim)).astype(
            np.float32
        )
        cache = dec_cross_attn_cache_torch[0]
        torch_out = (
            torch_layer(torch.as_tensor(x), torch_state, cross_attn_cache=cache)
            .detach()
            .numpy()
        )

    # JAX state
    rngs = nnx.Rngs(0)
    jax_layer = jax_layers.DecoderLayer(
        config=dia_config, compute_dtype=jnp.float32, rngs=rngs
    )
    jax_layer = convert_torch_to_nnx(torch_layer, jax_layer, dia_config=dia_config)
    jax_inference_state = jax_state.EncoderInferenceState.new(
        dia_config, jnp.ones((batch_size, 1, seq_len), dtype=jnp.int32)
    )
    # Create Jax decoder and precompute cross-attn cache
    jax_decoder = jax_layers.Decoder(
        config=dia_config, compute_dtype=jnp.float32, rngs=rngs
    )
    dec_cross_attn_cache = jax_decoder.precompute_cross_attn_cache(jnp.array(enc_out))
    jax_state_obj = jax_state.DecoderInferenceState.new(
        dia_config,
        enc_state=jax_inference_state,
        enc_out=jnp.array(enc_out),
        dec_cross_attn_cache=dec_cross_attn_cache,
        compute_dtype=jnp.float32,
        max_generation_length=seq_len,
    )
    cache = dec_cross_attn_cache[0]
    jax_out = jax_layer(jnp.array(x), jax_state_obj, cross_attn_cache=cache)
    np.testing.assert_allclose(jax_out, torch_out, rtol=1e-4, atol=1e-4)


def test_decoder_equivalence():
    # Load config and weights
    dia = Dia.from_local(
        config_path=os.path.join(os.path.dirname(__file__), "../config.json"),
        checkpoint_path=os.path.join(
            os.path.dirname(__file__), "../weights/dia-v1.pth"
        ),
        compute_dtype="float32",
        device="cpu",
        load_dac=False,
    )
    dia_config = dia.config
    batch_size = 2
    seq_len = 4
    enc_embed_dim = dia_config.encoder_config.hidden_size
    torch_model = dia.model
    torch_decoder = torch_model.decoder
    torch_decoder.eval()

    # Prepare input and state
    enc_out = np.random.uniform(-1, 1, (batch_size * 2, seq_len, enc_embed_dim)).astype(
        np.float32
    )
    torch_inference_state = dia_state.EncoderInferenceState.new(
        dia_config, torch.ones((batch_size, 1, seq_len), dtype=torch.long)
    )
    dec_cross_attn_cache_torch = torch_decoder.precompute_cross_attn_cache(
        torch.as_tensor(enc_out)
    )
    torch_state = dia_state.DecoderInferenceState.new(
        dia_config,
        enc_state=torch_inference_state,
        enc_out=enc_out,
        dec_cross_attn_cache=dec_cross_attn_cache_torch,
        compute_dtype=torch.float32,
        max_generation_length=seq_len,
    )
    with torch.no_grad():
        # Input to forward has shape [B*2, T, C], for example [2, 953, 9]
        x = np.ones(
            (batch_size * 2, seq_len, dia_config.decoder_config.num_channels)
        ).astype(np.long)
        torch_out = torch_decoder(torch.as_tensor(x), torch_state).numpy()

    # JAX
    # NOTE: We need float64 precision because the error blows up in all the layers
    rngs = nnx.Rngs(0)
    jax_decoder = jax_layers.Decoder(
        config=dia_config, compute_dtype=jnp.float64, rngs=rngs
    )
    jax_decoder = convert_torch_to_nnx(
        torch_decoder, jax_decoder, dia_config=dia_config
    )
    jax_inference_state = jax_state.EncoderInferenceState.new(
        dia_config, jnp.ones((batch_size, 1, seq_len), dtype=jnp.int64)
    )
    dec_cross_attn_cache = jax_decoder.precompute_cross_attn_cache(jnp.array(enc_out))
    jax_state_obj = jax_state.DecoderInferenceState.new(
        dia_config,
        enc_state=jax_inference_state,
        enc_out=jnp.array(enc_out).astype(jnp.float64),
        dec_cross_attn_cache=dec_cross_attn_cache,
        compute_dtype=jnp.float64,
        max_generation_length=seq_len,
    )
    jax_out = jax_decoder(jnp.array(x), jax_state_obj)
    np.testing.assert_allclose(jax_out, torch_out, rtol=1e-4, atol=1e-4)
