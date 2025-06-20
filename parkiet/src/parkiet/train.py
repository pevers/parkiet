import os
import math
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from parkiet.dia.model import ComputeDtype, Dia
from parkiet.dia.state import DecoderOutput
from parkiet.dia.training.state import EncoderTrainingState, DecoderTrainingState


class TrainingConfig:
    def __init__(self, **kwargs):
        self.batch_size: int = kwargs.get("batch_size", 1)
        self.learning_rate: float = kwargs.get("learning_rate", 2e-3)
        self.num_epochs: int = kwargs.get("num_epochs", 500)
        self.warmup_steps: int = kwargs.get("warmup_steps", 2000)
        self.total_steps: int = kwargs.get("total_steps", 110_000)
        self.text_dropout_prob: float = kwargs.get("text_dropout_prob", 0.10)
        self.gradient_accumulation_steps: int = kwargs.get(
            "gradient_accumulation_steps", 1
        )
        self.checkpoint_dir: str = kwargs.get("checkpoint_dir", "weights")
        self.checkpoint_every: int = kwargs.get("checkpoint_every", 50)
        self.device: str = kwargs.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )


def get_cosine_schedule_with_warmup(
    optimizer: optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def compute_loss(
    dia: Dia,
    text_tokens: torch.Tensor,  # [B, T]
    audio_tokens: torch.Tensor,  # [B, S, C]
    text_dropout_prob: float = 0.10,
) -> Dict[str, torch.Tensor]:
    """Compute the loss for a batch of data.

    Args:
        model: The Dia model
        text_tokens: Text token ids [batch_size, text_seq_len]
        audio_tokens: Audio token ids [batch_size, audio_seq_len, channels]
        text_dropout_prob: Probability of dropping out text condition for CFG training

    Returns:
        Dictionary containing loss values and metrics
    """
    model = dia.model
    batch_size = text_tokens.shape[0]
    device = text_tokens.device

    # Classifier free guidance: drop text tokens for a percentage of samples in the batch
    dropout_mask = torch.rand(batch_size, device=device) < text_dropout_prob
    text_tokens = torch.where(
        dropout_mask.unsqueeze(1),  # [B, 1] to broadcast across sequence length
        torch.zeros_like(text_tokens),  # unconditional (no text)
        text_tokens,  # conditional (with text)
    )
    enc_state = EncoderTrainingState.new(model.config, text_tokens)
    encoder_outputs = model.encoder(text_tokens, enc_state)

    # Get decoder outputs
    dec_cross_attn_cache = model.decoder.precompute_cross_attn_cache(
        encoder_outputs, enc_state.positions, enc_state.padding_mask
    )
    dec_state = DecoderTrainingState.new(
        model.config,
        enc_state,
        encoder_outputs,
        dec_cross_attn_cache,
        dia.compute_dtype,
        max_generation_length=audio_tokens.shape[1],
    )
    prefill, prefill_steps = dia._prepare_audio_prompt(audio_tokens)
    dec_output = DecoderOutput.new(batch_size, model.config, device)
    dec_output.prefill(prefill, prefill_steps)

    dec_step = min(prefill_steps) - 1
    if dec_step > 0:
        dec_state.prepare_step(0, dec_step)
        output_tokens = dec_output.get_tokens_at(0, dec_step)
        model.decoder.forward(output_tokens, dec_state)

    # Get decoder outputs
    decoder_outputs = model.decoder(output_tokens, dec_state)

    # Calculate loss for multi-channel outputs - only ignore padding tokens
    # Reshape logits and targets for cross-entropy calculation
    # decoder_outputs: [batch_size, seq_len, channels, vocab_size]
    # decoder_target: [batch_size, seq_len, channels]
    vocab_size = decoder_outputs.shape[-1]
    loss = nn.CrossEntropyLoss(ignore_index=model.config.data.audio_pad_value)(
        decoder_outputs.view(-1, vocab_size),  # [B*T*C, V]
        audio_tokens.view(-1),  # [B*T*C]
    )

    return {
        "loss": loss,
    }


def train_epoch(
    dia: Dia,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    config: TrainingConfig,
    epoch: int,
) -> Dict[str, float]:
    """Train for one epoch."""
    model = dia.model
    total_loss = 0
    total_steps = 0

    for batch_idx, batch in enumerate(train_loader):
        text_tokens = batch["text"].to(config.device)
        audio_tokens = batch["audio"].to(config.device)
        loss_dict = compute_loss(
            dia,
            text_tokens,
            audio_tokens,
            config.text_dropout_prob,
        )
        loss = loss_dict["loss"] / config.gradient_accumulation_steps
        loss.backward()

        if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()

        total_loss += loss.item() * config.gradient_accumulation_steps
        total_steps += 1

        if batch_idx % 100 == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch}, Step {batch_idx}, Loss: {loss.item():.4f}, LR: {lr:.6f}"
            )

    return {
        "loss": total_loss / total_steps,
    }


def main():
    training_config = TrainingConfig()
    dia = Dia.from_local(
        config_path="config.json",
        checkpoint_path="weights/dia-v0_1.pth",
        compute_dtype=ComputeDtype.BFLOAT16,
        device=training_config.device,
        load_dac=True,
    )
    model = dia.model

    # Initialize optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )

    # Initialize learning rate scheduler
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=training_config.warmup_steps,
        num_training_steps=training_config.total_steps,
    )

    # Initialize data loader
    from parkiet.dataset import create_dataloader

    train_loader = create_dataloader(
        parquet_path="../data/chunks/chunks_dataset.parquet",
        config=dia.config,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
    )

    # Training loop
    model.train()
    for epoch in range(training_config.num_epochs):
        metrics = train_epoch(
            dia=dia,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            config=training_config,
            epoch=epoch,
        )

        print(f"Epoch {epoch} completed. Average loss: {metrics['loss']:.4f}")

        # Save checkpoint
        if (epoch + 1) % training_config.checkpoint_every == 0:
            os.makedirs(training_config.checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(
                training_config.checkpoint_dir, f"dia-v0_1-epoch{epoch + 1}.pth"
            )
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

    # Save final checkpoint after training is complete
    os.makedirs(training_config.checkpoint_dir, exist_ok=True)
    final_checkpoint_path = os.path.join(
        training_config.checkpoint_dir,
        f"dia-v0_1-final-epoch{training_config.num_epochs}.pth",
    )
    torch.save(model.state_dict(), final_checkpoint_path)
    print(f"Training completed. Saved final checkpoint to {final_checkpoint_path}")


if __name__ == "__main__":
    main()
