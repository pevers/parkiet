import argparse
import os
import torch

import jax.numpy as jnp
from parkiet.dia.model import Dia, ComputeDtype
from parkiet.jax.model import convert_torch_to_nnx
from parkiet.jax.layers import DiaModel


def main():
    parser = argparse.ArgumentParser(description="Convert PyTorch model to JAX format")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "../../config.json"),
        help="Path to config JSON file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "../../weights/dia-nl-v1.pth"),
        help="Path to PyTorch checkpoint file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "../../weights/jax-nl-v1"),
        help="Output path for JAX checkpoint",
    )
    args = parser.parse_args()

    dia = Dia.from_local(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        compute_dtype=ComputeDtype.BFLOAT16,
        device=torch.device("cpu"),
        load_dac=False,
    )
    dia.model.eval()
    dia_config = dia.config
    torch_model = dia.model

    import flax.nnx as nnx

    rngs = nnx.Rngs(0)
    jax_model = DiaModel(
        config=dia_config,
        param_dtype=jnp.float32,
        compute_dtype=jnp.bfloat16,
        rngs=rngs,
    )
    jax_model = convert_torch_to_nnx(torch_model, jax_model, dia_config=dia_config)
    _, state = nnx.split(jax_model)
    params = nnx.to_pure_dict(state)

    # Convert device arrays to host arrays for serialization
    import jax

    global_params = jax.device_get(params)

    # Checkpoint using Orbax
    import orbax.checkpoint as ocp

    with ocp.StandardCheckpointer() as ckptr:
        ckptr.save(args.output, global_params)

    print(f"JAX model saved to {args.output}")


if __name__ == "__main__":
    main()
