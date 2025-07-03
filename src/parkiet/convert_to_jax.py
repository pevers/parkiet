import os
import torch
from parkiet.dia.model import Dia, ComputeDtype
from parkiet.jax.model import convert_torch_to_nnx
from parkiet.jax.layers import DiaModel

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../../config.json")
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "../../weights/dia-v1.pth")
ORBAX_OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "../../weights/jax-v1")


def main():
    dia = Dia.from_local(
        config_path=CONFIG_PATH,
        checkpoint_path=CHECKPOINT_PATH,
        compute_dtype=ComputeDtype.FLOAT32,
        device=torch.device("cpu"),
        load_dac=False,
    )
    dia.model.eval()
    dia_config = dia.config
    torch_model = dia.model

    import flax.nnx as nnx

    rngs = nnx.Rngs(0)
    jax_model = DiaModel(
        config=dia_config, compute_dtype=ComputeDtype.FLOAT32, rngs=rngs
    )
    jax_model = convert_torch_to_nnx(torch_model, jax_model, dia_config=dia_config)
    _, state = nnx.split(jax_model)
    pure_dict = nnx.to_pure_dict(state)

    # Checkpoint using Orbax
    import orbax.checkpoint as ocp

    with ocp.StandardCheckpointer() as ckptr:
        ckptr.save(ORBAX_OUTPUT_PATH, pure_dict)


if __name__ == "__main__":
    main()
