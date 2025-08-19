"""
JAX training loop implementation using flax nnx.
"""

import logging
import os
import time
from pathlib import Path
from dataclasses import dataclass
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
from tqdm import tqdm
import orbax.checkpoint as ocp
from torch.utils.tensorboard import SummaryWriter
from parkiet.dia.config import DiaConfig
from parkiet.jax.model import Dia, ComputeDtype
from parkiet.jax.layers import DiaModel
from parkiet.torch.dataset import create_dataset
from parkiet.jax.training_state import (
    DecoderTrainingState,
    EncoderTrainingState,
    TrainingDecoderOutput,
)
from parkiet.jax.audio import build_delay_indices, apply_audio_delay

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)


@dataclass(frozen=True)
class JaxConfig:
    pad_token_id: int
    bos_token_id: int
    encoder_max_position_embeddings: int
    decoder_max_position_embeddings: int
    decoder_num_channels: int
    decoder_num_key_value_heads: int
    decoder_num_hidden_layers: int
    decoder_cross_head_dim: int
    compute_dtype: jnp.dtype
    delay_pattern: tuple[int, ...]


class TrainingConfig:
    """Configuration for training parameters."""

    def __init__(self, **kwargs):
        self.batch_size: int = kwargs.get("batch_size", 1)
        self.learning_rate: float = kwargs.get("learning_rate", 1e-4)
        self.warmup_steps: int = kwargs.get("warmup_steps", 100)
        self.total_steps: int = kwargs.get("total_steps", 2000)
        self.gradient_accumulation_steps: int = kwargs.get(
            "gradient_accumulation_steps", 8
        )
        self.checkpoint_dir: str = os.path.abspath(
            kwargs.get("checkpoint_dir", "weights")
        )
        self.checkpoint_every_steps: int = kwargs.get("checkpoint_every_steps", 500)
        self.sample_every_steps: int = kwargs.get("sample_every_steps", 200)
        self.log_every: int = kwargs.get("log_every", 10)
        self.max_grad_norm: float = kwargs.get("max_grad_norm", 1.0)
        self.log_dir: str = kwargs.get("log_dir", "logs")


def create_cosine_schedule_with_warmup(
    learning_rate: float,
    warmup_steps: int,
    total_steps: int,
    num_cycles: float = 0.5,
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


def compute_loss(
    model: DiaModel,
    text_tokens: jnp.ndarray,
    audio_tokens: jnp.ndarray,
    jax_config: JaxConfig,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """
    Compute the loss for a batch of data using teacher forcing.

    Args:
        model: The DiaModel instance
        text_tokens: Text token ids [batch_size, text_seq_len]
        audio_tokens: Audio token ids [batch_size, audio_seq_len, channels]
        jax_config: JAX configuration

    Returns:
        Tuple of (loss, metrics_dict)
    """
    batch_size = text_tokens.shape[0]
    audio_pad_value = jax_config.pad_token_id

    # Encode text
    enc_state = EncoderTrainingState.new(
        jax_config.encoder_max_position_embeddings, text_tokens
    )
    encoder_outputs = model.encoder(text_tokens, enc_state)

    # Precompute cross-attention cache
    dec_cross_attn_cache = model.decoder.precompute_cross_attn_cache(encoder_outputs)

    # Create decoder state
    actual_seq_len = audio_tokens.shape[1]
    dec_state = DecoderTrainingState.new(
        jax_config,
        enc_state,
        encoder_outputs,
        dec_cross_attn_cache,
        model.compute_dtype,
        max_generation_length=actual_seq_len,
    )

    # Apply delay pattern to audio tokens
    audio_seq_len = audio_tokens.shape[1]
    num_channels = jax_config.decoder_num_channels

    # Build delay indices for the input
    delay_indices = build_delay_indices(
        batch_size, audio_seq_len, num_channels, jax_config.delay_pattern
    )

    # Apply delay pattern to create shifted input
    audio_input = apply_audio_delay(
        audio_tokens,
        pad_value=jax_config.pad_token_id,
        bos_value=jax_config.bos_token_id,
        precomp=delay_indices,
    )

    # Forward pass through decoder
    dec_output = TrainingDecoderOutput.new(
        batch_size,
        jax_config.decoder_max_position_embeddings,
        jax_config.decoder_num_channels,
    )
    dec_output.prefill(audio_input, [audio_input.shape[1]] * batch_size)

    dec_step = audio_input.shape[1] - 1
    dec_state.prepare_step(0, dec_step)
    tokens = dec_output.get_tokens_at(0, dec_step)
    decoder_outputs = model.decoder(tokens, dec_state)

    # Compute loss
    # decoder_outputs: [batch_size, seq_len, channels, vocab_size]
    # target: [batch_size, seq_len, channels]
    vocab_size = decoder_outputs.shape[-1]

    # Create target tokens by shifting the delayed input tokens by 1 position
    # The decoder sees delayed_input[t] and should predict delayed_input[t+1]
    audio_target = jnp.concatenate(
        [
            audio_input[:, 1:, :],  # Delayed input shifted by 1
            jnp.full(
                (batch_size, 1, num_channels),
                jax_config.pad_token_id,
                dtype=audio_input.dtype,
            ),  # Add EOS/PAD
        ],
        axis=1,
    )

    # Ensure target sequence length matches decoder output
    decoder_seq_len = decoder_outputs.shape[1]
    audio_target = audio_target[:, :decoder_seq_len, :]

    # Reshape for cross-entropy loss
    logits = decoder_outputs.reshape(-1, vocab_size)  # [B*S*C, V]
    targets = audio_target.reshape(-1)  # [B*S*C]

    # Create mask for valid tokens (ignore padding)
    mask = targets != audio_pad_value

    # Compute cross-entropy loss
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    one_hot_targets = jax.nn.one_hot(targets, vocab_size)
    loss_per_token = -jnp.sum(log_probs * one_hot_targets, axis=-1)

    # Apply mask and compute mean loss
    masked_loss = loss_per_token * mask
    total_loss = jnp.sum(masked_loss)
    total_tokens = jnp.sum(mask)

    # Avoid division by zero
    loss = jnp.where(total_tokens > 0, total_loss / total_tokens, 0.0)

    # Compute accuracy
    predictions = jnp.argmax(logits, axis=-1)
    correct = (predictions == targets) * mask
    accuracy = jnp.where(total_tokens > 0, jnp.sum(correct) / total_tokens, 0.0)

    metrics = {
        "loss": loss,
        "accuracy": accuracy,
        "total_tokens": total_tokens,
    }

    return loss, metrics


@nnx.jit(static_argnums=(3,))  # jax_config is static
def train_step(
    model: DiaModel,
    optimizer: nnx.Optimizer,
    batch: dict[str, jnp.ndarray],
    jax_config: JaxConfig,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """
    Perform a single training step.

    Args:
        model: The model
        optimizer: The optimizer
        batch: Batch of data
        pad_token_id: Padding token ID
        bos_token_id: Beginning of sequence token ID
        max_text_len: Maximum text length
        num_channels: Number of audio channels
        compute_dtype: Computation dtype (static)

    Returns:
        Tuple of (loss, metrics)
    """

    def loss_fn(model):
        return compute_loss(model, batch["text"], batch["audio"], jax_config)

    # Compute gradients
    (loss, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)

    # Update parameters
    optimizer.update(grads)

    return loss, metrics


def evaluate_step(
    model: DiaModel,
    batch: dict[str, jnp.ndarray],
    jax_config: JaxConfig,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """
    Perform a single evaluation step.

    Args:
        model: The model
        batch: Batch of data
        pad_token_id: Padding token ID
        bos_token_id: Beginning of sequence token ID
        max_text_len: Maximum text length
        num_channels: Number of audio channels
        compute_dtype: Computation dtype

    Returns:
        Tuple of (loss, metrics)
    """
    return compute_loss(model, batch["text"], batch["audio"], jax_config)


def save_checkpoint(
    model: DiaModel,
    step: int,
    checkpoint_dir: str,
):
    """
    Save model checkpoint using orbax.

    Args:
        model: The model to save
        step: Current training step
        checkpoint_dir: Directory to save checkpoint
    """
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    # Get model state
    _, state = nnx.split(model)
    state_dict = nnx.to_pure_dict(state)

    # Save checkpoint
    checkpoint_file = checkpoint_path / f"checkpoint_{step}"
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(checkpoint_file, state_dict)

    log.info(f"Saved checkpoint to {checkpoint_file}")


def main():
    """Main training function."""
    training_config = TrainingConfig()

    # Load model configuration
    dia_config = DiaConfig.load("config.test.json")
    if dia_config is None:
        raise ValueError("Failed to load model configuration")
    compute_dtype = ComputeDtype.BFLOAT16

    # Initialize model
    log.info("Initializing model...")
    # Load existing Dia weights
    # dia = Dia.from_local(
    #     config_path="config.json",
    #     checkpoint_path=(Path("weights") / "jax-v1").resolve().as_posix(),
    #     compute_dtype=compute_dtype,
    #     load_dac=False,  # Don't load DAC for training
    # )
    dia = Dia(
        config=dia_config,
        compute_dtype=compute_dtype,
        load_dac=False,  # Don't load DAC for training
    )
    # Create JAX-compatible config
    jax_config = JaxConfig(
        pad_token_id=dia_config.pad_token_id,
        bos_token_id=dia_config.bos_token_id,
        encoder_max_position_embeddings=dia_config.encoder_config.max_position_embeddings,
        decoder_max_position_embeddings=dia_config.decoder_config.max_position_embeddings,
        decoder_num_channels=dia_config.decoder_config.num_channels,
        decoder_num_key_value_heads=dia_config.decoder_config.num_key_value_heads,
        decoder_num_hidden_layers=dia_config.decoder_config.num_hidden_layers,
        decoder_cross_head_dim=dia_config.decoder_config.cross_head_dim,
        # Convert to tuple to make it hashable
        delay_pattern=tuple(dia_config.delay_pattern),
        compute_dtype=compute_dtype.to_dtype(),
    )
    model = dia.model

    # Initialize optimizer
    log.info("Initializing optimizer...")
    schedule = create_cosine_schedule_with_warmup(
        learning_rate=training_config.learning_rate,
        warmup_steps=training_config.warmup_steps,
        total_steps=training_config.total_steps,
    )

    tx = optax.chain(
        optax.clip_by_global_norm(training_config.max_grad_norm),
        optax.adamw(
            learning_rate=schedule,
            b1=0.9,
            b2=0.95,
            weight_decay=0.01,
        ),
    )

    optimizer = nnx.Optimizer(model, tx)

    # Initialize data loader
    log.info("Initializing data loader...")
    dataset = create_dataset(
        parquet_path="chunks_dataset.test.parquet",
        config=dia_config,
    )
    train_loader = dataset.batch_iterator(
        batch_size=training_config.batch_size,
        shuffle=True,
        use_weighted_sampling=True,
    )

    # Initialize TensorBoard writer
    log.info("Initializing TensorBoard writer...")
    writer = SummaryWriter(log_dir=training_config.log_dir)

    # Log configuration
    writer.add_text("Config/Training", str(vars(training_config)))

    # Training loop
    log.info("Starting training...")
    train_iter = iter(train_loader)

    # Initialize metrics
    running_loss = 0.0
    running_accuracy = 0.0

    pbar = tqdm(range(training_config.total_steps), desc="Training")

    for step in pbar:
        step_start_time = time.time()

        # Accumulate gradients
        batch_loss = 0.0
        batch_accuracy = 0.0

        for _ in range(training_config.gradient_accumulation_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            # Convert numpy arrays to JAX arrays
            jax_batch = {
                "text": jnp.array(batch["text"]),
                "audio": jnp.array(batch["audio"]),
            }

            # Training step
            loss, metrics = train_step(model, optimizer, jax_batch, jax_config)

            batch_loss += loss / training_config.gradient_accumulation_steps
            batch_accuracy += (
                metrics["accuracy"] / training_config.gradient_accumulation_steps
            )

        # Update running averages
        running_loss = 0.9 * running_loss + 0.1 * batch_loss
        running_accuracy = 0.9 * running_accuracy + 0.1 * batch_accuracy

        # Log progress
        current_lr = schedule(step)
        pbar.set_postfix(
            {
                "loss": f"{running_loss:.4f}",
                "acc": f"{running_accuracy:.4f}",
                "lr": f"{current_lr:.6f}",
            }
        )

        # Log metrics
        if step % training_config.log_every == 0:
            step_time = time.time() - step_start_time
            log.info(
                f"Step {step}: loss={batch_loss:.4f}, acc={batch_accuracy:.4f}, "
                f"lr={current_lr:.6f}, time={step_time:.3f}s"
            )

            # Log to TensorBoard
            writer.add_scalar("Train/Loss", float(batch_loss), step)
            writer.add_scalar("Train/Accuracy", float(batch_accuracy), step)
            writer.add_scalar("Train/LearningRate", float(current_lr), step)
            writer.add_scalar("Train/RunningLoss", float(running_loss), step)
            writer.add_scalar("Train/RunningAccuracy", float(running_accuracy), step)
            writer.add_scalar("Train/StepTime", step_time, step)

        # Save checkpoint
        if step > 0 and step % training_config.checkpoint_every_steps == 0:
            save_checkpoint(model, step, training_config.checkpoint_dir)

    # Save final checkpoint
    save_checkpoint(model, training_config.total_steps, training_config.checkpoint_dir)

    # Close TensorBoard writer
    writer.close()

    log.info("Training completed!")


if __name__ == "__main__":
    main()
