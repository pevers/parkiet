from typing import Iterator
import os
import jax

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update(
    "jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir"
)
jax.distributed.initialize()

from parkiet.jax.audio import apply_audio_delay, build_delay_indices

from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import wandb
from parkiet.dia.config import DiaConfig
from parkiet.jax.training_state import (
    DecoderTrainingState,
    EncoderTrainingState,
)
from parkiet.jax.dataset import (
    create_dataset,
    discover_parquet_shards,
    get_total_samples_from_shards,
    calculate_total_steps,
)
import jax.numpy as jnp
from datetime import datetime
import logging
import optax
import flax.nnx as nnx
from pathlib import Path
import numpy as np
import time
from parkiet.jax.layers import DiaModel
from parkiet.jax.tensorboard_logger import DistributedTensorBoardLogger
import orbax.checkpoint as ocp

# Set up logging fircst
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - [PID:%(process)d] - %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)

# Enable JAX compilation logging
jax.config.update("jax_log_compiles", True)
jax_logger = logging.getLogger("jax")
jax_logger.setLevel(logging.WARNING)


def make_mesh_dp_mp():
    """Create mesh for data and model parallel sharding."""
    procs = jax.process_count()  # 4 for t4-32
    local = jax.local_device_count()  # 4 for t4-32
    total_devices = jax.device_count()

    logger.info(
        f"Creating mesh: processes={procs}, local_devices={local}, total={total_devices}"
    )
    assert procs * local == total_devices, (
        f"Device count mismatch: {procs} * {local} != {total_devices}"
    )

    # For data and model parallel
    devs = np.array(jax.devices()).reshape(procs, local)
    mesh = Mesh(devs, axis_names=("data", "model"))
    logger.info(f"Created mesh with shape {mesh.shape} and axes {mesh.axis_names}")
    return mesh


mesh = make_mesh_dp_mp()


class TrainingConfig:
    """Configuration for training parameters."""

    def __init__(self, **kwargs):
        self.batch_size: int = kwargs.get("batch_size", 4)
        # Learning rate is small because we are fine-tuning on an existing (English) model
        self.learning_rate: float = kwargs.get("learning_rate", 4e-5)
        self.warmup_steps: int = kwargs.get("warmup_steps", 2000)
        self.total_epochs: int = kwargs.get("total_epochs", 3)
        # GA is high because the TPUs are not big enough for a larger batch size
        self.gradient_accumulation_steps: int = kwargs.get(
            "gradient_accumulation_steps", 32
        )
        self.checkpoint_dir: str = kwargs.get(
            "checkpoint_dir", "gs://parkiet-training/weights"
        )
        self.checkpoint_every_steps: int = kwargs.get("checkpoint_every_steps", 3000)
        self.sample_every_steps: int = kwargs.get("sample_every_steps", 100)
        self.log_every: int = kwargs.get("log_every", 100)
        self.max_grad_norm: float = kwargs.get("max_grad_norm", 1.0)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_log_dir = f"logs/parkiet-{timestamp}"
        self.log_dir: str = kwargs.get("log_dir", default_log_dir)


def load_checkpoint(checkpoint_path: str) -> dict:
    """Load checkpoint outside of JIT."""
    with ocp.StandardCheckpointer() as checkpointer:
        restored_params = checkpointer.restore(checkpoint_path)
    return restored_params


@nnx.jit(static_argnames=("dia_config", "compute_dtype", "param_dtype"))
def create_model(
    dia_config: DiaConfig,
    compute_dtype: jnp.dtype = jnp.float32,
    param_dtype: jnp.dtype = jnp.float32,
    restored_params: dict | None = None,
    rngs: nnx.Rngs = nnx.Rngs(0),
) -> DiaModel:
    """Create a model without sharding."""
    model = DiaModel(
        dia_config, compute_dtype=compute_dtype, param_dtype=param_dtype, rngs=rngs
    )

    if restored_params is not None:
        graphdef, state = nnx.split(model)
        nnx.replace_by_pure_dict(state, restored_params)
        model = nnx.merge(graphdef, state)

    # TODO: This is probably not needed as the model is already sharded
    state = nnx.state(model)
    pspecs = nnx.get_partition_spec(state)
    sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
    nnx.update(model, sharded_state)
    return model


def save_distributed_checkpoint(
    model: DiaModel,
    step: int,
    checkpoint_dir: str,
) -> None:
    """Save a distributed checkpoint to Google Cloud Storage."""
    checkpoint_path = f"{checkpoint_dir}/checkpoint_{step:06d}"
    sharded_state = nnx.state(model)
    pure_dict_state = nnx.to_pure_dict(sharded_state)
    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(checkpoint_path, pure_dict_state)
    checkpointer.wait_until_finished()


@nnx.jit(
    static_argnames=("learning_rate", "warmup_steps", "total_steps", "max_grad_norm")
)
def create_optimizer(
    model: DiaModel,
    learning_rate: float,
    warmup_steps: int,
    total_steps: int,
    max_grad_norm: float = 1.0,
) -> nnx.Optimizer:
    """Create an optimizer with gradient clipping and learning rate schedule."""
    schedule = create_cosine_schedule_with_warmup(
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        num_cycles=1.0,
    )

    # Create optimizer chain
    tx = optax.chain(
        optax.zero_nans(),
        optax.clip_by_global_norm(max_grad_norm),
        optax.adamw(
            learning_rate=schedule,
            b1=0.9,
            b2=0.95,
            weight_decay=0.01,
        ),
    )

    optimizer = nnx.Optimizer(model, tx)
    return optimizer


def create_cosine_schedule_with_warmup(
    learning_rate: float,
    warmup_steps: int,
    total_steps: int,
    num_cycles: float = 0.25,
) -> optax.Schedule:
    """Create a cosine learning rate schedule with warmup."""

    def schedule_fn(step):
        warmup_lr = learning_rate * step / warmup_steps

        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        cosine_lr = (
            learning_rate * 0.5 * (1.0 + jnp.cos(jnp.pi * num_cycles * 2.0 * progress))
        )

        return jnp.where(step < warmup_steps, warmup_lr, cosine_lr)

    return schedule_fn


@nnx.jit(static_argnames=("dia_config",))
def compute_loss(
    model: DiaModel,
    text_tokens: jnp.ndarray,
    audio_input: jnp.ndarray,
    audio_target: jnp.ndarray,
    dia_config: DiaConfig,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """
    Compute the loss for a batch of data using teacher forcing.

    Args:
        model: The DiaModel instance
        text_tokens: Text token ids [batch_size, text_seq_len]
        audio_input: Audio input tokens [batch_size, audio_seq_len, channels] (already delayed)
        audio_target: Audio target tokens [batch_size, audio_seq_len, channels] (calculated from delayed input)
        dia_config: DiaConfig object

    Returns:
        Tuple of (loss, metrics_dict)
    """
    batch_size = text_tokens.shape[0]

    # Encode text
    enc_state = EncoderTrainingState.new(
        dia_config.encoder_config.max_position_embeddings, text_tokens
    )
    encoder_outputs = model.encoder(text_tokens, enc_state)

    # Create simplified decoder state for training
    audio_seq_len = audio_input.shape[1]
    dec_state = DecoderTrainingState.new(
        dia_config,
        enc_state,
        encoder_outputs,
        model.compute_dtype,
        max_generation_length=audio_seq_len,
    )

    # Forward pass through decoder in training mode (no cache management)
    decoder_outputs = model.decoder(audio_input, dec_state)

    # Compute loss
    # decoder_outputs: [batch_size, seq_len, channels, vocab_size]
    # audio_target: [batch_size, seq_len, channels]
    vocab_size = decoder_outputs.shape[-1]

    # Reshape for cross-entropy loss
    logits = decoder_outputs.reshape(-1, vocab_size)  # [B*T*C, V]
    num_channels = dia_config.decoder_config.num_channels
    channel_weights_3d = jnp.tile(
        jnp.where(jnp.arange(num_channels) == 0, 4.0, 1.0)[None, None, :],
        (batch_size, audio_target.shape[1], 1),
    )

    targets = audio_target.reshape(-1)  # [B*T*C]
    channel_weights_flat = channel_weights_3d.reshape(-1)  # [B*T*C]

    # Exclude padding token from loss
    meaningful_mask = targets != dia_config.pad_token_id

    # Compute cross-entropy loss only for non-padding tokens
    loss_per_token = optax.softmax_cross_entropy(
        logits, jax.nn.one_hot(targets, vocab_size)
    )

    # Apply mask and channel weights - only meaningful tokens
    valid_loss = loss_per_token * meaningful_mask * channel_weights_flat

    # Compute mean only over meaningful tokens (excluding padding)
    total_valid_weight = jnp.sum(meaningful_mask * channel_weights_flat)
    loss = jnp.sum(valid_loss) / jnp.maximum(total_valid_weight, 1e-8)

    # Compute accuracy only for meaningful tokens
    predictions = jnp.argmax(logits, axis=-1)
    correct = (predictions == targets) * meaningful_mask
    accuracy = jnp.sum(correct) / jnp.maximum(jnp.sum(meaningful_mask), 1.0)

    # Debugging metrics
    total_tokens = targets.shape[0]
    non_pad_tokens = jnp.sum(meaningful_mask)
    pad_ratio = non_pad_tokens / total_tokens
    avg_masked_loss_per_token = jnp.sum(valid_loss) / jnp.maximum(non_pad_tokens, 1.0)

    # Channel-specific debugging metrics
    # Create channel indices for debugging
    targets_reshaped = targets.reshape(batch_size, -1, num_channels)  # [B, T, C]
    channel_indices = jnp.tile(
        jnp.arange(num_channels)[None, None, :],
        (batch_size, targets_reshaped.shape[1], 1),
    ).reshape(-1)  # [B*T*C]

    channel_losses = {}
    channel_accuracies = {}
    for ch in range(num_channels):
        ch_mask = (channel_indices == ch) * meaningful_mask
        ch_sum = jnp.sum(ch_mask)
        channel_losses[f"channel_{ch}_loss"] = jnp.where(
            ch_sum > 0,
            jnp.sum(loss_per_token * ch_mask) / jnp.maximum(ch_sum, 1.0),
            0.0,
        )
        channel_accuracies[f"channel_{ch}_acc"] = jnp.where(
            ch_sum > 0,
            jnp.sum((predictions == targets) * ch_mask) / jnp.maximum(ch_sum, 1.0),
            0.0,
        )

    metrics = {
        "loss": loss,
        "accuracy": accuracy,
        "pad_ratio": pad_ratio,
        "avg_masked_loss": avg_masked_loss_per_token,
        "non_pad_tokens": non_pad_tokens,
        "total_tokens": total_tokens,
        **channel_losses,
        **channel_accuracies,
    }

    return loss, metrics


@nnx.jit(static_argnames=("dia_config",))
def compute_gradients_step(
    model: DiaModel, batch: dict[str, jnp.ndarray], dia_config: DiaConfig
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray], dict]:
    """Compute gradients for a single batch without updating optimizer."""

    def loss_fn(m):
        return compute_loss(
            m,
            batch["text"],
            batch["audio_input"],
            batch["audio_target"],
            dia_config,
        )

    (loss, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    return loss, metrics, grads


@nnx.jit(static_argnames=("num_accumulation_steps"))
def apply_accumulated_gradients(
    optimizer: nnx.Optimizer, accumulated_grads: dict, num_accumulation_steps: int
) -> None:
    """Apply accumulated gradients scaled by the number of accumulation steps."""
    # Scale gradients by the number of accumulation steps
    scaled_grads = jax.tree_map(lambda g: g / num_accumulation_steps, accumulated_grads)
    optimizer.update(scaled_grads)


def create_dataloader(
    dataset,
    batch_size: int,
) -> Iterator[dict[str, jnp.ndarray]]:
    """
    Create a dataloader using the provided dataset.

    Args:
        dataset: AudioTextDataset instance
        batch_size: Batch size

    Returns:
        Iterator over batches
    """
    logger.info(f"Creating dataloader with batch size: {batch_size}")
    logger.info(f"Dataset has {len(dataset.parquet_files)} shards")

    # Use the dataset's batch iterator
    for batch in dataset.batch_iterator(
        batch_size=batch_size, shuffle=True, seed=42, use_sample_prob=True
    ):
        batch_jax = {k: jnp.array(v) for k, v in batch.items()}
        yield batch_jax


@nnx.vmap(in_axes=(0, None))
def prepare_input_target_pair(
    audio_sequence: jnp.ndarray, pad_token_id: int
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Prepare input and target from a single BOS,audio_tokens,EOS sequence.

    Args:
        audio_sequence: Audio sequence of shape [seq_len, C] with BOS,audio_tokens,EOS
        pad_token_id: Padding token ID

    Returns:
        Tuple of (input_sequence, target_sequence) both of shape [seq_len, C]
        - input: BOS,audio_tokens,PAD
        - target: audio_tokens,EOS,PAD
    """
    # Input: same as input sequence (BOS,audio_tokens,EOS,PAD...)
    audio_input = audio_sequence

    # Target: shift left by 1 (audio_tokens,EOS,PAD,PAD...)
    audio_target = jnp.concatenate(
        [
            audio_sequence[1:],  # audio_tokens,EOS,PAD...
            jnp.full(
                (1, audio_sequence.shape[1]), pad_token_id, dtype=audio_sequence.dtype
            ),  # Add PAD at end
        ],
        axis=0,
    )

    return audio_input, audio_target


def load_and_prepare_batch(
    batch_iterator: Iterator[dict[str, jnp.ndarray]], dia_config: DiaConfig
) -> dict[str, jnp.ndarray]:
    """
    Load batch and prepare input/target.

    Args:
        batch_iterator: Iterator over batches
        dia_config: DiaConfig object

    Returns:
        Batch ready for training
    """
    # Load batch data
    raw_batch = next(batch_iterator)

    # Apply delay pattern to the original audio sequence first
    batch_size = raw_batch["audio"].shape[0]
    delay_precomp = build_delay_indices(
        B=batch_size,
        T=raw_batch["audio"].shape[1],
        C=raw_batch["audio"].shape[2],
        delay_pattern=dia_config.delay_pattern,
    )

    # Apply delay pattern to get delayed input
    delayed_audio = apply_audio_delay(
        audio_BxTxC=jnp.array(raw_batch["audio"]),
        pad_value=dia_config.pad_token_id,
        bos_value=dia_config.bos_token_id,
        precomp=delay_precomp,
    )

    # Prepare input and target sequences from the delayed audio
    audio_input, audio_target = prepare_input_target_pair(
        delayed_audio, dia_config.pad_token_id
    )

    # Create complete batch
    batch = {
        "text": raw_batch["text"],
        "audio_input": audio_input,
        "audio_target": audio_target,
    }

    return batch


def evaluate_step(
    model: DiaModel, batch: dict[str, jnp.ndarray], dia_config: DiaConfig
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """
    Perform a single evaluation step.

    Args:
        model: The model
        batch: Batch of data
        dia_config: DiaConfig object

    Returns:
        Tuple of (loss, metrics)
    """
    return compute_loss(
        model,
        batch["text"],
        batch["audio_input"],
        batch["audio_target"],
        dia_config,
    )


def main():
    # Get device and process info
    devices = jax.devices()
    num_devices = len(devices)
    process_id = jax.process_index()
    num_processes = jax.process_count()

    compute_dtype = jnp.bfloat16

    # config.json = 1.5B original model
    # config.tiny.json = much smaller model for testing
    dia_config_frz = DiaConfig.load("config.json")
    logger.info(f"Config: {dia_config_frz}")
    logger.info(f"Process {process_id} of {num_processes}")
    logger.info(f"Device count: {num_devices}")
    logger.info(f"Devices: {[d.id for d in devices]}")

    # Initialize wandb if token is available
    wandb_api_key = os.getenv("WANDB_API_KEY")
    use_wandb = wandb_api_key is not None
    run = None

    if use_wandb:
        logger.info("WANDB_API_KEY found, initializing wandb logging")
        run = wandb.init(
            project="parkiet-jax-distributed",
            name=f"jax-distributed-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "architecture": "Dia",
                "dataset": "AudioText",
                "epochs": 6,
            },
        )
    else:
        logger.info("No WANDB_API_KEY found, skipping wandb logging")

    # Load checkpoint outside the mesh context (before JIT)
    # This needs to happen on all processes
    checkpoint_path = (Path("weights") / "jax-v1").resolve().as_posix()
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    restored_params = load_checkpoint(checkpoint_path)

    with mesh:
        # Create sharded model with same rng key for all processes
        # Use a deterministic seed for all processes to ensure consistent initialization
        rngs = nnx.Rngs(42)
        model = create_model(
            dia_config_frz,
            param_dtype=jnp.float32,
            compute_dtype=compute_dtype,
            restored_params=restored_params,
            rngs=rngs,
        )
        training_config = TrainingConfig()

        # Discover parquet shards and calculate dataset size first
        parquet_files = discover_parquet_shards("shards")
        logger.info(f"Found {len(parquet_files)} parquet shards total")

        total_samples = get_total_samples_from_shards(parquet_files)
        logger.info(f"Total samples in dataset: {total_samples:,}")

        total_steps = calculate_total_steps(
            total_samples=total_samples,
            batch_size=training_config.batch_size,
            gradient_accumulation_steps=training_config.gradient_accumulation_steps,
            total_epochs=training_config.total_epochs,
        )
        logger.info(f"Total training steps: {total_steps:,}")

        logger.info("Creating optimizer...")
        optimizer = create_optimizer(
            model=model,
            learning_rate=training_config.learning_rate,
            warmup_steps=training_config.warmup_steps,
            total_steps=total_steps,
            max_grad_norm=training_config.max_grad_norm,
        )

        # Create schedule separately
        lr_schedule = create_cosine_schedule_with_warmup(
            learning_rate=training_config.learning_rate,
            warmup_steps=training_config.warmup_steps,
            total_steps=total_steps,
        )

        # Set checkpoint directory (GCS path)
        checkpoint_dir = training_config.checkpoint_dir

        # Initialize TensorBoard logger
        tb_logger = DistributedTensorBoardLogger(training_config.log_dir, process_id)

        logger.info("Starting distributed training loop...")

        # Calculate steps per epoch
        steps_per_epoch = total_steps // training_config.total_epochs
        logger.info(f"Steps per epoch: {steps_per_epoch}")

        if steps_per_epoch == 0:
            logger.warning(
                "Steps per epoch is 0! This will cause immediate epoch completion."
            )
            steps_per_epoch = max(1, total_steps)  # Ensure at least 1 step per epoch

        # Create dataset once outside the epoch loop
        logger.info("Creating dataset...")
        dataset = create_dataset(
            config=dia_config_frz,
            parquet_path="shards",
            transcription_clean_prob=0.1,
            text_dropout_prob=0.15,
        )

        step = 0
        for epoch in range(training_config.total_epochs):
            logger.info(f"Starting epoch {epoch}")

            # Reset dataset for new epoch
            dataset.reset()

            # Create dataloader
            batch_iterator = create_dataloader(
                dataset,
                training_config.batch_size,
            )

            epoch_step = 0
            while epoch_step < steps_per_epoch:
                # Start timing the step
                step_start_time = time.time()

                # Accumulate loss and metrics over gradient_accumulation_steps
                accumulated_loss = 0.0
                accumulated_metrics = {}
                accumulated_grads = None

                # Track if we successfully completed all gradient accumulation steps
                completed_accumulation_steps = 0

                for _ in range(training_config.gradient_accumulation_steps):
                    try:
                        # Load and prepare batch
                        batch = load_and_prepare_batch(batch_iterator, dia_config_frz)
                        batch = jax.device_put(batch, NamedSharding(mesh, P("data")))
                        completed_accumulation_steps += 1

                        # Compute gradients
                        loss, metrics, grads = compute_gradients_step(
                            model, batch, dia_config_frz
                        )

                        # Accumulate gradients
                        if accumulated_grads is None:
                            accumulated_grads = grads
                        else:
                            accumulated_grads = jax.tree_map(
                                lambda acc_g, g: acc_g + g, accumulated_grads, grads
                            )

                        accumulated_loss += float(loss)
                        # Accumulate all metrics dynamically
                        for key, value in metrics.items():
                            if key not in accumulated_metrics:
                                accumulated_metrics[key] = 0.0
                            accumulated_metrics[key] += float(value)

                    except StopIteration:
                        logger.info(f"Dataset exhausted at step {step}, epoch {epoch}")
                        if completed_accumulation_steps == 0:
                            break
                        break

                # Skip this step if we couldn't load any batches
                if completed_accumulation_steps == 0:
                    break

                # Apply accumulated gradients (scaled by actual accumulation steps)
                apply_accumulated_gradients(
                    optimizer,
                    accumulated_grads,
                    completed_accumulation_steps,
                )

                # Average the metrics
                avg_loss = accumulated_loss / completed_accumulation_steps
                avg_metrics = {
                    key: val / completed_accumulation_steps
                    for key, val in accumulated_metrics.items()
                }

                # Log to TensorBoard
                tb_logger.log_scalar("train/loss", avg_loss, step)
                for key, val in avg_metrics.items():
                    tb_logger.log_scalar(f"train/{key}", val, step)

                # Calculate step time
                step_end_time = time.time()
                step_time = step_end_time - step_start_time

                # Log current learning rate
                current_lr = lr_schedule(step)
                tb_logger.log_scalar("train/learning_rate", float(current_lr), step)

                # Log to wandb if available
                if use_wandb and run is not None:
                    wandb_metrics = {
                        "train/loss": avg_loss,
                        "train/learning_rate": current_lr,
                        "train/step_time": step_time,
                    }
                    wandb_metrics.update(
                        {f"train/{k}": v for k, v in avg_metrics.items()}
                    )
                    run.log(wandb_metrics)

                if step % training_config.log_every == 0:
                    # Log channel-specific losses to diagnose imbalance
                    ch_loss_str = ", ".join(
                        [
                            f"Ch{i}: {avg_metrics.get(f'channel_{i}_loss', 0):.3f}"
                            for i in range(3)
                        ]
                    )  # Show first 3 channels
                    logger.info(
                        f"Step {step}, Loss: {avg_loss:.4f}, Accuracy: {avg_metrics['accuracy']:.4f}, "
                        f"Pad ratio: {avg_metrics['pad_ratio']:.4f}, Step time: {step_time:.2f}s, Channel losses: {ch_loss_str}"
                    )
                    tb_logger.flush()

                if step % training_config.checkpoint_every_steps == 0 and step > 0:
                    save_distributed_checkpoint(model, step, checkpoint_dir)
                    logger.info(f"Saved checkpoint at step {step}")

                step += 1
                epoch_step += 1

    # Close TensorBoard logger
    tb_logger.close()
    logger.info("Training completed")

    if use_wandb and run is not None:
        run.finish()


if __name__ == "__main__":
    main()
