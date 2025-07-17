"""
JAX training loop implementation using flax nnx.
"""

import logging
import math
import time
from pathlib import Path
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
from tqdm import tqdm
import orbax.checkpoint as ocp

from parkiet.dia.config import DiaConfig
from parkiet.jax.model import Dia, ComputeDtype
from parkiet.jax.layers import DiaModel
from parkiet.jax.dataset import create_dummy_dataloader
from parkiet.jax.state import (
    DecoderTrainingState,
    EncoderTrainingState,
    DecoderOutput,
    DecoderInferenceState,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)


class TrainingConfig:
    """Configuration for training parameters."""

    def __init__(self, **kwargs):
        self.batch_size: int = kwargs.get("batch_size", 8)
        self.learning_rate: float = kwargs.get("learning_rate", 1e-4)
        self.warmup_steps: int = kwargs.get("warmup_steps", 100)
        self.total_steps: int = kwargs.get("total_steps", 2000)
        self.gradient_accumulation_steps: int = kwargs.get(
            "gradient_accumulation_steps", 8
        )
        self.checkpoint_dir: str = kwargs.get("checkpoint_dir", "weights_jax")
        self.checkpoint_every_steps: int = kwargs.get("checkpoint_every_steps", 500)
        self.sample_every_steps: int = kwargs.get("sample_every_steps", 500)
        self.log_every: int = kwargs.get("log_every", 10)
        self.max_grad_norm: float = kwargs.get("max_grad_norm", 1.0)


def create_cosine_schedule_with_warmup(
    learning_rate: float,
    warmup_steps: int,
    total_steps: int,
    num_cycles: float = 0.5,
) -> optax.Schedule:
    """Create a cosine learning rate schedule with warmup."""

    def schedule_fn(step):
        if step < warmup_steps:
            return learning_rate * step / warmup_steps

        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return (
            learning_rate * 0.5 * (1.0 + jnp.cos(jnp.pi * num_cycles * 2.0 * progress))
        )

    return schedule_fn


def compute_loss(
    model: DiaModel,
    text_tokens: jnp.ndarray,
    audio_tokens: jnp.ndarray,
    config: DiaConfig,
    compute_dtype: jnp.dtype,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """
    Compute the loss for a batch of data using teacher forcing.

    Args:
        model: The DiaModel instance
        text_tokens: Text token ids [batch_size, text_seq_len]
        audio_tokens: Audio token ids [batch_size, audio_seq_len, channels]
        config: Model configuration
        compute_dtype: Computation dtype

    Returns:
        Tuple of (loss, metrics_dict)
    """
    batch_size = text_tokens.shape[0]
    audio_pad_value = config.pad_token_id

    # Pad text input if needed
    max_text_len = config.encoder_config.max_position_embeddings
    if text_tokens.shape[1] < max_text_len:
        padding = jnp.zeros(
            (batch_size, max_text_len - text_tokens.shape[1]), dtype=text_tokens.dtype
        )
        text_tokens = jnp.concatenate([text_tokens, padding], axis=1)

    # Encode text
    enc_state = EncoderTrainingState.new(config, text_tokens)
    encoder_outputs = model.encoder(text_tokens, enc_state)  # type: ignore

    # Precompute cross-attention cache
    dec_cross_attn_cache = model.decoder.precompute_cross_attn_cache(encoder_outputs)

    # Create decoder state
    actual_seq_len = audio_tokens.shape[1]
    dec_state = DecoderTrainingState.new(
        config,
        enc_state,
        encoder_outputs,
        dec_cross_attn_cache,
        compute_dtype,
        max_generation_length=actual_seq_len,
    )

    # Prepare audio input (add BOS token)
    audio_bos_value = config.bos_token_id
    bos_tokens = jnp.full(
        (batch_size, 1, config.decoder_config.num_channels),
        audio_bos_value,
        dtype=audio_tokens.dtype,
    )
    audio_input = jnp.concatenate([bos_tokens, audio_tokens[:, :-1, :]], axis=1)

    # Forward pass through decoder
    dec_output = DecoderOutput.new(batch_size, config)
    dec_output.prefill(audio_input, [audio_input.shape[1]] * batch_size)

    dec_step = audio_input.shape[1] - 1
    dec_state.prepare_step(0, dec_step)
    tokens = dec_output.get_tokens_at(0, dec_step)

    # Create inference state with same data but compatible type
    inference_state = DecoderInferenceState(
        dtype=dec_state.dtype,
        enc_out=dec_state.enc_out,
        enc_positions=dec_state.enc_positions,
        dec_positions=dec_state.dec_positions,
        self_attn_cache=dec_state.self_attn_cache
        or dec_state.cross_attn_cache,  # Use cross_attn_cache as fallback
        cross_attn_cache=dec_state.cross_attn_cache,
        causal_attn_mask=dec_state.causal_attn_mask,
        cross_attn_mask=dec_state.cross_attn_mask,
    )

    decoder_outputs = model.decoder(tokens, inference_state)

    # Compute loss
    # decoder_outputs: [batch_size, seq_len, channels, vocab_size]
    # target: [batch_size, seq_len, channels]
    vocab_size = decoder_outputs.shape[-1]

    # Reshape for cross-entropy loss
    logits = decoder_outputs.reshape(-1, vocab_size)  # [B*S*C, V]
    targets = audio_tokens.reshape(-1)  # [B*S*C]

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


def train_step(
    model: DiaModel,
    optimizer: nnx.Optimizer,
    batch: dict[str, jnp.ndarray],
    config: DiaConfig,
    compute_dtype: jnp.dtype,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """
    Perform a single training step.

    Args:
        model: The model
        optimizer: The optimizer
        batch: Batch of data
        config: Model configuration
        compute_dtype: Computation dtype

    Returns:
        Tuple of (loss, metrics)
    """

    def loss_fn(model):
        return compute_loss(model, batch["text"], batch["audio"], config, compute_dtype)

    # Compute gradients
    (loss, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)

    # Update parameters
    optimizer.update(grads)

    return loss, metrics


def evaluate_step(
    model: DiaModel,
    batch: dict[str, jnp.ndarray],
    config: DiaConfig,
    compute_dtype: jnp.dtype,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """
    Perform a single evaluation step.

    Args:
        model: The model
        batch: Batch of data
        config: Model configuration
        compute_dtype: Computation dtype

    Returns:
        Tuple of (loss, metrics)
    """
    return compute_loss(model, batch["text"], batch["audio"], config, compute_dtype)


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
    graphdef, state = nnx.split(model)
    state_dict = nnx.to_pure_dict(state)

    # Save checkpoint
    checkpoint_file = checkpoint_path / f"checkpoint_{step}"
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(checkpoint_file, state_dict)

    log.info(f"Saved checkpoint to {checkpoint_file}")


def main():
    """Main training function."""
    # Configuration
    training_config = TrainingConfig()

    # Load model configuration
    dia_config = DiaConfig.load("config.test.json")
    if dia_config is None:
        raise ValueError("Failed to load model configuration")
    compute_dtype = ComputeDtype.BFLOAT16

    # Initialize model
    log.info("Initializing model...")
    rngs = nnx.Rngs(0)
    dia = Dia(
        config=dia_config,
        compute_dtype=compute_dtype.name,
        load_dac=False,  # Don't load DAC for training
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
    train_loader = create_dummy_dataloader(
        config=dia_config,
        batch_size=training_config.batch_size,
        num_batches=training_config.total_steps
        * training_config.gradient_accumulation_steps,
        seed=42,
    )

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

        for acc_step in range(training_config.gradient_accumulation_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            # Training step
            loss, metrics = train_step(
                model, optimizer, batch, dia_config, compute_dtype.to_dtype()
            )

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

        # Save checkpoint
        if step > 0 and step % training_config.checkpoint_every_steps == 0:
            save_checkpoint(model, step, training_config.checkpoint_dir)

    # Save final checkpoint
    save_checkpoint(model, training_config.total_steps, training_config.checkpoint_dir)

    log.info("Training completed!")


if __name__ == "__main__":
    main()
