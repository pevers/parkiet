import jax.numpy as jnp
from flax.traverse_util import flatten_dict, unflatten_dict
import flax.nnx as nnx
from flax.nnx.statelib import replace_by_pure_dict
import logging

from parkiet.dia.config import DiaConfig

log = logging.getLogger(__name__)


def torch_to_flax_weight(torch_tensor):
    # Convert torch tensor to numpy, then to jax array
    return jnp.array(torch_tensor.cpu().numpy())


def convert_torch_to_nnx(torch_model, nnx_model, dia_config: DiaConfig):
    """
    Map PyTorch model weights to nnx model weights.
    """
    torch_state = torch_model.state_dict()
    graphdef, state = nnx.split(nnx_model)
    state_dict = nnx.to_pure_dict(state)
    state_dict = flatten_dict(state_dict)
    new_flat = {}
    for k, v in state_dict.items():
        torch_key = ".".join([str(s) for s in k])
        torch_key = torch_key.replace("embedding.embedding", "embedding.weight")
        torch_key = torch_key.replace("pre_sa_norm.scale", "pre_sa_norm.weight")
        torch_key = torch_key.replace("post_sa_norm.scale", "post_sa_norm.weight")
        torch_key = torch_key.replace("pre_ca_norm.scale", "pre_ca_norm.weight")
        torch_key = torch_key.replace("pre_mlp_norm.scale", "pre_mlp_norm.weight")
        torch_key = torch_key.replace("q_proj.kernel", "q_proj.weight")
        torch_key = torch_key.replace("k_proj.kernel", "k_proj.weight")
        torch_key = torch_key.replace("v_proj.kernel", "v_proj.weight")
        torch_key = torch_key.replace("o_proj.kernel", "o_proj.weight")
        torch_key = torch_key.replace("c_proj.kernel", "c_proj.weight")
        torch_key = torch_key.replace("wi_fused.kernel", "wi_fused.weight")
        torch_key = torch_key.replace("wo.kernel", "wo.weight")
        torch_key = torch_key.replace("norm.scale", "norm.weight")
        torch_key = torch_key.replace("logits_dense.kernel", "logits_dense.weight")

        # There is probably a better/faster way but it works and it is easy to understand
        for i in range(0, dia_config.decoder_config.num_channels):
            torch_key = torch_key.replace(
                f"embeddings.{i}.embedding", f"embeddings.{i}.weight"
            )

        found = False
        for tk in torch_state:
            if tk == torch_key:
                arr = torch_state[tk]
                new_flat[k] = torch_to_flax_weight(arr)
                found = True
                break
        if not found:
            # TODO: Fix timescale variables, they should not be part of the state
            log.warning(f"Weight {k} not found in PyTorch state")

    new_params = unflatten_dict(new_flat)
    replace_by_pure_dict(state, new_params)
    return nnx.merge(graphdef, state)
