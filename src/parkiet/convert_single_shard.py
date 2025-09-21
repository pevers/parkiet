# Converts the checkpoint on a single CPU / GPU machine from a sharded multi-device checkpoint to a single-device checkpoint

import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=16"

import argparse
import jax
import orbax.checkpoint as ocp
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Convert sharded checkpoint to single-device checkpoint"
    )
    parser.add_argument(
        "--checkpoint_path", required=True, help="Path to the sharded checkpoint"
    )
    parser.add_argument(
        "--output_path", required=True, help="Output path for the unsharded checkpoint"
    )

    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint_path).resolve()
    output_path = Path(args.output_path).resolve()

    checkpointer = ocp.StandardCheckpointer()
    restored_state = checkpointer.restore(checkpoint_path)

    state_host = jax.tree_util.tree_map(
        lambda x: jax.device_get(x) if isinstance(x, jax.Array) else x,
        restored_state,
    )
    checkpointer.save(output_path, state_host)
    checkpointer.wait_until_finished()


if __name__ == "__main__":
    main()
