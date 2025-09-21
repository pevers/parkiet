import argparse
import torch

import jax.numpy as jnp
from parkiet.dia.model import Dia, ComputeDtype
from parkiet.jax.model import convert_jax_to_torch
from parkiet.jax.layers import DiaModel
from parkiet.dia.config import DiaConfig
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Convert JAX model to PyTorch format")
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Path to config JSON file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="weights/jax-v1",
        help="Path to JAX checkpoint directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="weights/dia-v1-from-jax.pth",
        help="Output path for PyTorch checkpoint",
    )

    args = parser.parse_args()

    # Load config
    dia_config = DiaConfig.load(args.config)
    if dia_config is None:
        raise FileNotFoundError(f"Config file not found at {args.config}")

    # Create JAX model and load from orbax checkpoint
    import flax.nnx as nnx
    import orbax.checkpoint as ocp

    rngs = nnx.Rngs(0)
    jax_model = DiaModel(
        config=dia_config,
        param_dtype=jnp.float32,
        compute_dtype=jnp.float32,
        rngs=rngs,
    )

    # Load JAX checkpoint
    with ocp.StandardCheckpointer() as checkpointer:
        restored_params = checkpointer.restore(Path(args.checkpoint).resolve())

    # Update JAX model parameters
    graphdef, state = nnx.split(jax_model)
    nnx.replace_by_pure_dict(state, restored_params)
    jax_model = nnx.merge(graphdef, state)

    # Create PyTorch model
    torch_dia = Dia(
        config=dia_config,
        compute_dtype=ComputeDtype.FLOAT32,
        device=torch.device("cpu"),
        load_dac=False,
    )
    torch_model = torch_dia.model

    # Convert JAX to PyTorch
    torch_model = convert_jax_to_torch(jax_model, torch_model, dia_config)

    # Save PyTorch checkpoint
    torch.save(torch_model.state_dict(), args.output)
    print(f"PyTorch model saved to {args.output}")


if __name__ == "__main__":
    main()
