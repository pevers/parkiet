import os
import torch
import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn
from parkiet.dia.config import DiaConfig
from parkiet.dia.model import Dia
import parkiet.jax.state as jax_state
import parkiet.jax.layers as jax_layers
from parkiet.jax.model import convert_torch_to_flax

import parkiet.dia.layers as dia_layers
import parkiet.dia.state as dia_state

# For debugging
np.set_printoptions(precision=10, suppress=False)
torch.set_printoptions(
    precision=15,    # number of digits after the decimal point
    sci_mode=False,  # turn off scientific notation
)
jax.config.update("jax_enable_x64", True)


def test_dense_general_equivalence():
    x = np.ones((1, 128), dtype=np.float32)
    torch_dense_general = dia_layers.DenseGeneral(
        in_shapes=(128,), out_features=(128,), axis=(-1,), weight_dtype=torch.float32
    )
    torch_out = torch_dense_general(torch.as_tensor(x)).detach().numpy()
    jax_dense_general = nn.DenseGeneral(
        features=(128,), axis=(-1,), param_dtype=jnp.float32, use_bias=False
    )
    flax_params_loaded = {
        "kernel": jnp.array(torch_dense_general.weight.detach().numpy())
    }
    flax_out = jnp.asarray(
        jax_dense_general.apply({"params": flax_params_loaded}, jnp.array(x))
    )
    np.testing.assert_allclose(flax_out, torch_out, rtol=1e-4, atol=1e-4)

def test_mlp_block_equivalence():
    x = np.ones((1, 128), dtype=np.float32)
    torch_mlp_block = dia_layers.MlpBlock(
        embed_dim=128, intermediate_dim=256, compute_dtype=torch.float32
    )
    torch_mlp_block.eval()
    torch.nn.init.xavier_uniform(torch_mlp_block.wi_fused.weight)
    torch.nn.init.xavier_uniform(torch_mlp_block.wo.weight)
    torch_out = torch_mlp_block(torch.as_tensor(x)).detach().numpy()
    jax_mlp_block = jax_layers.MlpBlock(
        embed_dim=128, intermediate_dim=256, compute_dtype=jnp.float32
    )
    key = jax.random.PRNGKey(0)
    jax_mlp_block_vars = jax_mlp_block.init(key, jnp.array(x))
    params = jax_mlp_block_vars["params"]
    flax_params_loaded = convert_torch_to_flax(torch_mlp_block, params)
    flax_out = jnp.asarray(
        jax_mlp_block.apply({"params": flax_params_loaded}, jnp.array(x))
    )
    np.testing.assert_allclose(flax_out, torch_out, rtol=1e-5, atol=1e-5)


def test_rotary_embedding_equivalence():
    x = np.ones((1, 128), dtype=np.float64)
    positions = np.random.randint(1, 10_000, (1, 128), dtype=np.int32)
    positions = positions.astype(np.float64)
    torch_rotary_embedding = dia_layers.RotaryEmbedding(
        embedding_dims=128, min_timescale=1, max_timescale=10_000, dtype=torch.float64
    )
    torch_out = torch_rotary_embedding(
        torch.as_tensor(x), torch.as_tensor(positions)
    ).numpy()
    jax_rotary_embedding = jax_layers.RotaryEmbedding(
        embedding_dims=128, min_timescale=1, max_timescale=10_000, compute_dtype=jnp.float64
    )
    key = jax.random.PRNGKey(0)
    jax_rotary_embedding_vars = jax_rotary_embedding.init(
        key, jnp.array(x), jnp.array(positions)
    )
    flax_out = jnp.asarray(
        jax_rotary_embedding.apply(
            jax_rotary_embedding_vars, jnp.array(x), jnp.array(positions)
        )
    )
    np.testing.assert_allclose(flax_out, torch_out, rtol=1e-4, atol=1e-4)


def test_dot_product_attention_equivalence():
    # PyTorch expects: [batch, num_heads, seq_len, head_dim]
    # JAX expects: [batch, seq_len, num_heads, head_dim]
    query = np.random.uniform(-1, 1, (1, 1, 10, 2)).astype(np.float64)  # [batch, heads, seq_len, head_dim]
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

    jax_out = jax.nn.dot_product_attention(
        jnp.array(query_jax),
        jnp.array(key_jax),
        jnp.array(value_jax),
        bias=None,
        mask=None,
        scale=1.0,
        is_causal=False
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
    jax_rms = nn.RMSNorm(epsilon=1e-5, dtype=jnp.float32)
    key = jax.random.PRNGKey(0)
    jax_out = jax_rms.apply(jax_rms.init(key, jnp.array(x)), jnp.array(x))
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
        config=dia_config,
        q_embed_dim=1024,
        kv_embed_dim=1024,
        num_query_heads=16,
        num_kv_heads=16,
        head_dim=64,
        compute_dtype=torch.float32,
    )
    torch.nn.init.xavier_uniform(torch_self_attention.q_proj.weight)
    torch.nn.init.xavier_uniform(torch_self_attention.k_proj.weight)
    torch.nn.init.xavier_uniform(torch_self_attention.v_proj.weight)
    torch.nn.init.xavier_uniform(torch_self_attention.o_proj.weight)
    torch_out = (
        torch_self_attention(
            torch.as_tensor(X),
            torch.as_tensor(q_positions),
            torch.as_tensor(kv_positions),
            None,  # No mask
            None,  # No cache
        )
        .detach()
        .numpy()
    )
    jax_self_attention = jax_layers.SelfAttention(
        config=dia_config,
        q_embed_dim=1024,
        kv_embed_dim=1024,
        num_query_heads=16,
        num_kv_heads=16,
        head_dim=64,
        compute_dtype=jnp.float32,
    )
    key = jax.random.PRNGKey(0)
    jax_self_attention_vars = jax_self_attention.init(
        key, jnp.array(X), jnp.array(q_positions), jnp.array(kv_positions), None
    )
    params = jax_self_attention_vars["params"]
    flax_params_loaded = convert_torch_to_flax(torch_self_attention, params)
    flax_out = jnp.asarray(
        jax_self_attention.apply(
            {
                "params": flax_params_loaded,
                "constants": jax_self_attention_vars["constants"],
            },
            jnp.array(X),
            jnp.array(q_positions),
            jnp.array(kv_positions),
            None,
        )
    )
    np.testing.assert_allclose(flax_out, torch_out, rtol=1e-4, atol=1e-4)


def test_encoder_layer_equivalence():
    # Load config
    config = DiaConfig.load(os.path.join(os.path.dirname(__file__), "../config.json"))
    vocab_size = config.model.src_vocab_size
    seq_len = config.data.text_length

    input_tokens = torch.randint(0, vocab_size, (1, 1, seq_len), dtype=torch.long)
    torch_inference_state = dia_state.EncoderInferenceState.new(config, input_tokens)

    # PyTorch
    torch_layer = dia_layers.EncoderLayer(config, compute_dtype=torch.float32)
    torch_layer.eval()

    # Initialize weights
    for _, param in torch_layer.named_parameters():
        if param.dim() > 1:
            torch.nn.init.xavier_uniform(param)

    embedding = np.random.uniform(-1, 1, (2, 1024, 1024)).astype(np.float32)
    torch_out = torch_layer(torch.as_tensor(embedding), torch_inference_state).detach().numpy()

    # JAX
    jax_layer = jax_layers.EncoderLayer(config=config, compute_dtype=jnp.float32)
    key = jax.random.PRNGKey(0)
    jax_inference_state = jax_state.EncoderInferenceState.new(config, jnp.array(input_tokens))
    jax_vars = jax_layer.init(key, jnp.array(embedding), jax_inference_state)
    params = jax_vars["params"]
    flax_params_loaded = convert_torch_to_flax(torch_layer, params)
    flax_out = jnp.asarray(
        jax_layer.apply(
            {"params": flax_params_loaded, "constants": jax_vars["constants"]},
            jnp.array(embedding),
            jax_inference_state,
        )
    )
    np.testing.assert_allclose(flax_out, torch_out, rtol=1e-4, atol=1e-4)

def test_fused_qkv_equivalence():
    raise NotImplementedError("Fused QKV is not implemented in JAX")


def test_self_attention_cache():
    pass


def test_encoder_equivalence():
    # Load config and weights
    dia = Dia.from_local(
        config_path=os.path.join(os.path.dirname(__file__), "../config.json"),
        checkpoint_path=os.path.join(os.path.dirname(__file__), "../weights/dia-v0_1.pth"),
        compute_dtype="float32",
        device="cpu",
        load_dac=False,
    )

    # Instantiate PyTorch model and load weights
    config = dia.config
    torch_model = dia.model
    torch_model.load_state_dict(dia.model.state_dict())
    torch_encoder = torch_model.encoder
    torch_encoder.eval()

    # Prepare input and state
    seq_len = config.data.text_length
    input_tokens = torch.ones((1, 1, seq_len), dtype=torch.long)
    torch_inference_state = dia_state.EncoderInferenceState.new(config, input_tokens)

    # For CFG
    #encoder_input = input_tokens.squeeze(1).repeat_interleave(2, dim=0)
    embedding = np.random.uniform(-1, 1, (2, 1024, 1024)).astype(np.float32)

    # Run PyTorch encoder
    with torch.no_grad():
        torch_out = torch_encoder(torch.as_tensor(embedding), torch_inference_state).numpy()

    # Prepare Flax encoder and params
    flax_encoder = jax_layers.Encoder(config=config, compute_dtype=jnp.float64)
    # Flax params are initialized randomly; we need to convert and load from torch
    key = jax.random.PRNGKey(0)
    jax_input_tokens = jnp.array(input_tokens.numpy())
    jax_inference_state = jax_state.EncoderInferenceState.new(config, jax_input_tokens)
    #jax_encoder_input = jax_input_tokens.squeeze(1).repeat(2, axis=0)  # For CFG
    flax_vars = flax_encoder.init(key, jnp.array(embedding), jax_inference_state)
    params = flax_vars["params"]
    flax_params_loaded = convert_torch_to_flax(torch_encoder, params)

    # Run Flax encoder
    flax_out = jnp.asarray(
        flax_encoder.apply(
            {"params": flax_params_loaded, "constants": flax_vars["constants"]},
            jnp.array(embedding),
            jax_inference_state,
        )
    )

    # Compare outputs
    np.testing.assert_allclose(torch_out, flax_out, rtol=1e-4, atol=1e-4)
