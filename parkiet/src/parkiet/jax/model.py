import jax.numpy as jnp
from flax.core import freeze, unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict
from parkiet.dia.config import DiaConfig


def torch_to_flax_weight(torch_tensor):
    # Convert torch tensor to numpy, then to jax array
    return jnp.array(torch_tensor.cpu().numpy())


def convert_torch_to_flax(torch_model, flax_model_params):
    """
    Map PyTorch model weights to Flax model weights.
    This function assumes the architectures are identical and key names are similar.
    """
    torch_state = torch_model.state_dict()
    flax_params = unfreeze(flax_model_params)
    flat_flax = flatten_dict(flax_params)
    new_flat = {}
    for k, v in flat_flax.items():
        # Build the corresponding torch key

        # TODO: This is too aggresive! You are clearly doing something wrong in the model
        torch_key = ".".join(k)
        torch_key = torch_key.replace("embedding.embedding", "embedding.weight")
        torch_key = torch_key.replace("pre_sa_norm.scale", "pre_sa_norm.weight")
        torch_key = torch_key.replace("post_sa_norm.scale", "post_sa_norm.weight")
        torch_key = torch_key.replace("q_proj.kernel", "q_proj.weight")
        torch_key = torch_key.replace("k_proj.kernel", "k_proj.weight")
        torch_key = torch_key.replace("v_proj.kernel", "v_proj.weight")
        torch_key = torch_key.replace("o_proj.kernel", "o_proj.weight")
        torch_key = torch_key.replace("c_proj.kernel", "c_proj.weight")
        torch_key = torch_key.replace("wi_fused.kernel", "wi_fused.weight")
        torch_key = torch_key.replace("wo.kernel", "wo.weight")
        torch_key = torch_key.replace("norm.scale", "norm.weight")

        # Try to find the corresponding torch key
        found = False
        for tk in torch_state:
            if tk.endswith(torch_key):
                arr = torch_state[tk]
                new_flat[k] = torch_to_flax_weight(arr)
                found = True
                break
        if not found:
            raise ValueError(f"Weight {k} not found in PyTorch state")
    new_params = unflatten_dict(new_flat)
    print("Flax params loaded from torch:", new_params.keys())
    return freeze(new_params)
