"""
JAX training loop implementation using flax nnx.
"""

import logging
import os
import time
from datetime import datetime
from pathlib import Path
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
import orbax.checkpoint as ocp
import wandb
from torch.utils.tensorboard import SummaryWriter
from parkiet.dia.config import DiaConfig
from parkiet.jax.audio import apply_audio_delay, build_delay_indices
from parkiet.jax.layers import DiaModel
from parkiet.jax.dataset import create_dataset
from parkiet.jax.training_state import (
    DecoderTrainingState,
    EncoderTrainingState,
)
from parkiet.jax.dataset import (
    discover_parquet_shards,
    get_total_samples_from_shards,
    calculate_total_steps,
)

# Enable JAX compilation logging to monitor recompilation
jax.config.update("jax_log_compiles", True)
jax_logger = logging.getLogger("jax")
jax_logger.setLevel(logging.WARNING)


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
    batch_iterator, dia_config: DiaConfig
) -> dict[str, jnp.ndarray]:
    """
    Load batch and prepare input/target.

    Args:
        batch_iterator: Iterator over batches
        dia_config: DiaConfig object

    Returns:
        Batch ready for training
    """
    # Load raw batch
    raw_batch = next(batch_iterator)

    # Convert numpy arrays to JAX arrays
    batch_jax = {k: jnp.array(v) for k, v in raw_batch.items()}

    # Apply delay pattern to the original audio sequence first
    batch_size = batch_jax["audio"].shape[0]
    delay_precomp = build_delay_indices(
        B=batch_size,
        T=batch_jax["audio"].shape[1],
        C=batch_jax["audio"].shape[2],
        delay_pattern=dia_config.delay_pattern,
    )

    # Apply delay pattern to get delayed input
    delayed_audio = apply_audio_delay(
        audio_BxTxC=batch_jax["audio"],
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
        "text": batch_jax["text"],
        "audio_input": audio_input,
        "audio_target": audio_target,
    }

    return batch


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True
)
log = logging.getLogger(__name__)


class TrainingConfig:
    """Configuration for training parameters."""

    def __init__(self, **kwargs):
        self.batch_size: int = kwargs.get("batch_size", 1)
        self.learning_rate: float = kwargs.get("learning_rate", 2e-4)
        self.warmup_steps: int = kwargs.get("warmup_steps", 200)
        self.total_epochs: int = kwargs.get("total_epochs", 6)
        self.gradient_accumulation_steps: int = kwargs.get(
            "gradient_accumulation_steps", 32
        )
        self.checkpoint_dir: str = os.path.abspath(
            kwargs.get("checkpoint_dir", "weights")
        )
        self.checkpoint_every_steps: int = kwargs.get("checkpoint_every_steps", 2000)
        self.sample_every_steps: int = kwargs.get("sample_every_steps", 10)
        self.log_every: int = kwargs.get("log_every", 100)
        self.max_grad_norm: float = kwargs.get("max_grad_norm", 1.0)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_log_dir = f"logs/parkiet-{timestamp}"
        self.log_dir: str = kwargs.get("log_dir", default_log_dir)
        self.tensorboard_dir: str = kwargs.get(
            "tensorboard_dir", f"logs/parkiet-{timestamp}"
        )


def create_cosine_schedule_with_warmup(
    learning_rate: float,
    warmup_steps: int,
    total_steps: int,
    num_cycles: float = 1.0,
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

    # Audio input is already delayed from load_and_prepare_batch

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

    # Exclude both PAD and BOS tokens from loss
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
    model: DiaModel,
    batch: dict[str, jnp.ndarray],
    dia_config: DiaConfig,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray], dict]:
    """
    Compute gradients for a single batch without updating optimizer.

    Args:
        model: The model
        batch: Batch of data
        dia_config: DiaConfig object

    Returns:
        Tuple of (loss, metrics, grads)
    """

    def loss_fn(model):
        return compute_loss(
            model,
            batch["text"],
            batch["audio_input"],
            batch["audio_target"],
            dia_config,
        )

    # Compute gradients
    (loss, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)

    return loss, metrics, grads


@nnx.jit(static_argnames=("num_accumulation_steps",))
def apply_accumulated_gradients(
    optimizer: nnx.Optimizer, accumulated_grads: dict, num_accumulation_steps: int
) -> None:
    """Apply accumulated gradients scaled by the number of accumulation steps."""
    # Scale gradients by the number of accumulation steps
    scaled_grads = jax.tree_map(lambda g: g / num_accumulation_steps, accumulated_grads)
    optimizer.update(scaled_grads)


def evaluate_step(
    model: DiaModel,
    batch: dict[str, jnp.ndarray],
    dia_config: DiaConfig,
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


def save_checkpoint(
    model: DiaModel,
    step: int,
    checkpoint_dir: str,
):
    """
    Save model state to checkpoint.

    Args:
        model: The model
        step: Step number
        checkpoint_dir: Path to checkpoint directory
    """
    checkpoint_path = f"{checkpoint_dir}/checkpoint_{step:06d}"
    state = nnx.state(model)
    pure_dict_state = nnx.to_pure_dict(state)
    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(checkpoint_path, pure_dict_state)
    checkpointer.wait_until_finished()


def load_checkpoint(checkpoint_path: str) -> dict:
    """Load checkpoint outside of JIT."""
    checkpointer = ocp.StandardCheckpointer()
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
    """Create a model."""
    model = DiaModel(
        dia_config, compute_dtype=compute_dtype, param_dtype=param_dtype, rngs=rngs
    )

    if restored_params is not None:
        graphdef, state = nnx.split(model)
        nnx.replace_by_pure_dict(state, restored_params)
        model = nnx.merge(graphdef, state)

    return model


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


def main():
    """Main training function for single-process training."""
    # Load model configuration
    dia_config = DiaConfig.load("config.small.json")
    if dia_config is None:
        raise ValueError("Failed to load model configuration")

    log.info(f"Config: {dia_config}")

    compute_dtype = jnp.bfloat16
    param_dtype = jnp.float32

    # Load checkpoint if available
    # checkpoint_path = (Path("weights") / "jax-v1").resolve().as_posix()
    # restored_params = None
    # if Path(checkpoint_path).exists():
    #     log.info(f"Loading checkpoint from {checkpoint_path}")
    #     restored_params = load_checkpoint(checkpoint_path)

    # Initialize model
    log.info("Initializing model...")
    rngs = nnx.Rngs(42)
    model = create_model(
        dia_config,
        compute_dtype=compute_dtype,
        param_dtype=param_dtype,
        restored_params=None,
        rngs=rngs,
    )

    log.info("Model initialized successfully")

    training_config = TrainingConfig()

    # Discover parquet shards and calculate dataset size first
    parquet_files = discover_parquet_shards("shards")
    log.info(f"Found {len(parquet_files)} parquet shards")

    total_samples = get_total_samples_from_shards(parquet_files)
    log.info(f"Total samples in dataset: {total_samples:,}")

    total_steps = calculate_total_steps(
        total_samples=total_samples,
        batch_size=training_config.batch_size,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        total_epochs=training_config.total_epochs,
    )
    log.info(f"Total training steps: {total_steps:,}")

    # Initialize optimizer
    log.info("Initializing optimizer...")
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

    # Set checkpoint directory
    checkpoint_dir = training_config.checkpoint_dir

    # Initialize TensorBoard writer
    os.makedirs(training_config.tensorboard_dir, exist_ok=True)
    tb_writer = SummaryWriter(training_config.tensorboard_dir)
    log.info(f"TensorBoard logging initialized at: {training_config.tensorboard_dir}")

    # Initialize wandb if token is available
    wandb_api_key = os.getenv("WANDB_API_KEY")
    use_wandb = wandb_api_key is not None
    run = None

    if use_wandb:
        log.info("WANDB_API_KEY found, initializing wandb logging")
        run = wandb.init(
            project="parkiet-jax-training",
            name=f"jax-train-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "architecture": "Dia",
                "dataset": "AudioText",
                "epochs": training_config.total_epochs,
            },
        )
    else:
        log.info("No WANDB_API_KEY found, skipping wandb logging")

    log.info("Starting training loop...")

    # Calculate steps per epoch
    steps_per_epoch = total_steps // training_config.total_epochs
    log.info(f"Steps per epoch: {steps_per_epoch}")

    if steps_per_epoch == 0:
        log.warning("Steps per epoch is 0! This will cause immediate epoch completion.")
        steps_per_epoch = max(1, total_steps)  # Ensure at least 1 step per epoch

    # Create dataset once outside the epoch loop
    log.info("Creating dataset...")
    dataset = create_dataset(
        config=dia_config,
        parquet_path="shards",
        transcription_clean_prob=0.1,
        text_dropout_prob=0.15,
    )

    step = 0
    for epoch in range(training_config.total_epochs):
        log.info(f"Starting epoch {epoch}")

        # Reset dataset for new epoch
        dataset.reset()

        # Create batch iterator for this epoch
        batch_iterator = dataset.batch_iterator(
            batch_size=training_config.batch_size,
            shuffle=True,
            seed=42,
            use_sample_prob=False,
        )

        epoch_step = 0
        while epoch_step < steps_per_epoch:
            step_start_time = time.time()

            # Accumulate loss and metrics over gradient_accumulation_steps
            accumulated_loss = 0.0
            accumulated_metrics = {}
            accumulated_grads = None

            # Track if we successfully completed all gradient accumulation steps
            completed_accumulation_steps = 0

            for _ in range(training_config.gradient_accumulation_steps):
                try:
                    batch = load_and_prepare_batch(batch_iterator, dia_config)
                    completed_accumulation_steps += 1

                    # Compute gradients
                    loss, metrics, grads = compute_gradients_step(
                        model, batch, dia_config
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
                    log.info(f"Dataset exhausted at step {step}, epoch {epoch}")
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

            # Calculate step time
            step_end_time = time.time()
            step_time = step_end_time - step_start_time

            # Log current learning rate
            current_lr = lr_schedule(step)

            # Log to TensorBoard (convert JAX arrays to numpy)
            tb_writer.add_scalar("train/loss", float(avg_loss), step)
            tb_writer.add_scalar("train/learning_rate", float(current_lr), step)
            tb_writer.add_scalar("train/step_time", float(step_time), step)
            for k, v in avg_metrics.items():
                tb_writer.add_scalar(f"train/{k}", float(v), step)

            # Log to wandb if available
            if use_wandb and run is not None:
                wandb_metrics = {
                    "train/loss": avg_loss,
                    "train/learning_rate": current_lr,
                    "train/step_time": step_time,
                }
                wandb_metrics.update({f"train/{k}": v for k, v in avg_metrics.items()})
                run.log(wandb_metrics)

            if step % training_config.log_every == 0:
                # Log channel-specific losses to diagnose imbalance
                ch_loss_str = ", ".join(
                    [
                        f"Ch{i}: {avg_metrics.get(f'channel_{i}_loss', 0):.3f}"
                        for i in range(3)
                    ]
                )  # Show first 3 channels
                log.info(
                    f"Step {step}, Loss: {avg_loss:.4f}, Accuracy: {avg_metrics['accuracy']:.4f}, "
                    f"Pad ratio: {avg_metrics['pad_ratio']:.4f}, "
                    f"Step time: {step_time:.2f}s, Channel losses: {ch_loss_str}"
                )

            if step % training_config.checkpoint_every_steps == 0 and step > 0:
                save_checkpoint(
                    model,
                    step,
                    checkpoint_dir,
                )
                log.info(f"Saved checkpoint at step {step}")

            step += 1
            epoch_step += 1

        log.info(f"Completed epoch {epoch}. Steps in epoch: {epoch_step}")

    log.info("Training completed")
    tb_writer.close()
    if use_wandb and run is not None:
        run.finish()


if __name__ == "__main__":
    main()
