import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from parkiet.dia.config import DiaConfig
from parkiet.dia.model import ComputeDtype, Dia
from parkiet.dia.state import DecoderOutput
from parkiet.dia.training.state import DecoderTrainingState, EncoderTrainingState


class TrainingConfig:
    def __init__(self, **kwargs):
        self.batch_size: int = kwargs.get("batch_size", 8)
        self.learning_rate: float = kwargs.get("learning_rate", 1e-4)
        self.warmup_steps: int = kwargs.get("warmup_steps", 100)
        self.total_steps: int = kwargs.get("total_steps", 2000)
        self.gradient_accumulation_steps: int = kwargs.get(
            "gradient_accumulation_steps", 8
        )
        self.checkpoint_dir: str = kwargs.get("checkpoint_dir", "weights")
        self.checkpoint_every_steps: int = kwargs.get("checkpoint_every_steps", 500)
        self.sample_every_steps: int = kwargs.get("sample_every_steps", 500)
        self.device: str = kwargs.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.log_dir: str = kwargs.get("log_dir", "logs")
        self.log_every: int = kwargs.get("log_every", 10)  # Log every N steps


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
) -> dict[str, torch.Tensor]:
    """Compute the loss for a batch of data using teacher forcing with delay pattern.

    Args:
        dia: The Dia model
        text_tokens: Text token ids [batch_size, text_seq_len]
        audio_tokens: Audio token ids [batch_size, audio_seq_len, channels]
        audio_target: Target audio tokens [batch_size, audio_seq_len, channels]
        steps: Prefill steps for each audio prompt [batch_size]
    Returns:
        Dictionary containing loss values and metrics
    """

    # TODO: We need to dropout 15% of the text samples in the batch

    device = dia.device
    model = dia.model
    batch_size = text_tokens.shape[0]
    audio_pad_value = model.config.data.audio_pad_value

    # Pad text input, encoding is already done
    text_tokens = dia._pad_text_input(text_tokens).squeeze(1)

    # Encode text
    enc_state = EncoderTrainingState.new(model.config, text_tokens)
    encoder_outputs = model.encoder(text_tokens, enc_state)

    # Precompute cross-attention cache
    dec_cross_attn_cache = model.decoder.precompute_cross_attn_cache(
        encoder_outputs, enc_state.positions, enc_state.padding_mask
    )

    # Create decoder state
    actual_seq_len = audio_tokens.shape[1]
    dec_state = DecoderTrainingState.new(
        model.config,
        enc_state,
        encoder_outputs,
        dec_cross_attn_cache,
        dia.compute_dtype,
        max_generation_length=actual_seq_len,
    )

    prefill, prefill_steps = dia._prepare_audio_prompt(audio_tokens)
    dec_output = DecoderOutput.new(batch_size, model.config, device)
    dec_output.prefill(prefill, prefill_steps)

    dec_step = min(prefill_steps) - 1
    dec_state.prepare_step(0, dec_step)
    tokens = dec_output.get_tokens_at(0, dec_step)
    decoder_outputs = model.decoder.forward(tokens, dec_state)

    # Output of the audio is the delayed input shifted by one with EOS
    audio_target = tokens[:, 1:, :]
    eos_token = torch.full(
        (batch_size, 1, model.config.data.channels),
        model.config.data.audio_eos_value,
        device=device,
    )
    audio_target = torch.cat([audio_target, eos_token], dim=1)

    # Calculate loss
    # decoder_outputs: [batch_size, actual_seq_len, channels, vocab_size]
    # target: [batch_size, actual_seq_len, channels]
    vocab_size = decoder_outputs.shape[-1]

    # Get logits for the full sequence prediction
    logits = decoder_outputs.reshape(-1, vocab_size)  # [B*S*C, V]

    # The target is the input shifted by one
    targets = audio_target.reshape(-1).to(dtype=torch.long)  # [B*S*C]

    # Calculate cross-entropy loss, ignoring padding tokens
    loss_fn = nn.CrossEntropyLoss(ignore_index=audio_pad_value)
    loss = loss_fn(logits, targets)

    return {
        "loss": loss,
    }


def evaluate_and_generate_sample(
    dia: Dia, step: int, prompt: str, output_dir: str = "output_samples"
):
    """Generate a sample from the model and save it to a file."""
    model = dia.model
    model.eval()

    print(f"\nGenerating sample for step {step} with prompt: '{prompt}'")
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        audio = dia.generate(
            text=prompt,
            temperature=0.0,
            cfg_scale=0.0,
        )
        output_path = os.path.join(output_dir, f"sample_step_{step}.wav")
        dia.save_audio(output_path, audio)
        print(f"Saved sample to {output_path}\n")

    model.train()


def main():
    training_config = TrainingConfig()

    # Create TensorBoard writer
    os.makedirs(training_config.log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=training_config.log_dir)

    dia_config = DiaConfig.load("config.test.json")
    dia = Dia(
        config=dia_config,
        compute_dtype=ComputeDtype.BFLOAT16,
        device=training_config.device,
        load_dac=True,
    )
    dia.model.to(training_config.device)
    dia._load_dac_model()

    # dia = Dia.from_local(
    #     config_path="config.json",
    #     checkpoint_path="weights/dia-v0_1.pth",
    #     compute_dtype=ComputeDtype.BFLOAT16,
    #     device=training_config.device,
    #     load_dac=True,
    # )
    model = dia.model

    # Log model configuration to TensorBoard
    writer.add_text("Config/Model", str(dia.config))
    writer.add_text("Config/Training", str(vars(training_config)))

    # Try freezing all layers except the output projection, will only work for English models but might validate the training loop
    # for n, p in model.named_parameters():
    #     p.requires_grad = n.startswith("decoder.logits_dense")

    # Initialize optimizer
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=training_config.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )

    # Initialize data loader
    from parkiet.dataset import create_dataloader, create_test_dataloader

    train_loader = create_dataloader(
        parquet_path="./data/chunks_dataset.parquet",
        config=dia.config,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        max_text_length=dia.config.data.text_length,
    )

    dummy_loader = create_test_dataloader(
        dia=dia,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
    )

    total_training_steps = training_config.total_steps

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=training_config.warmup_steps,
        num_training_steps=total_training_steps,
    )

    # Training loop
    model.train()
    global_step = 0

    train_iter = iter(dummy_loader)
    pbar = tqdm(initial=global_step, total=total_training_steps, desc="Training")

    while global_step < total_training_steps:
        optimizer.zero_grad()

        batch_loss = 0.0
        for _ in range(training_config.gradient_accumulation_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(dummy_loader)
                batch = next(train_iter)
            text_tokens = batch["text"].to(training_config.device)
            audio_tokens = batch["audio"].to(training_config.device)
            loss_dict = compute_loss(dia, text_tokens, audio_tokens)
            loss = loss_dict["loss"] / training_config.gradient_accumulation_steps
            loss.backward()
            batch_loss += loss.item()

        # Calculate gradient norm
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1.0 / 2)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        global_step += 1
        pbar.update(1)

        lr = optimizer.param_groups[0]["lr"]
        pbar.set_postfix(loss=f"{batch_loss:.4f}", lr=f"{lr:.6f}")

        if writer is not None and global_step % training_config.log_every == 0:
            writer.add_scalar("Train/Loss", batch_loss, global_step)
            writer.add_scalar("Train/LearningRate", lr, global_step)
            writer.add_scalar("Train/GradientNorm", total_norm, global_step)

        if global_step > 0 and global_step % training_config.sample_every_steps == 0:
            evaluate_and_generate_sample(
                dia=dia,
                step=global_step,
                prompt="[S1] Hey, did you see that movie today?",
                output_dir="output_samples",
            )

        # Save checkpoint
        if (
            global_step > 0
            and global_step % training_config.checkpoint_every_steps == 0
        ):
            os.makedirs(training_config.checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(
                training_config.checkpoint_dir, f"dia-v0_1-step{global_step}.pth"
            )
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

    pbar.close()

    # Save final checkpoint after training is complete
    os.makedirs(training_config.checkpoint_dir, exist_ok=True)
    final_checkpoint_path = os.path.join(
        training_config.checkpoint_dir,
        f"dia-v0_1-final-step{global_step}.pth",
    )
    torch.save(model.state_dict(), final_checkpoint_path)
    print(f"Training completed. Saved final checkpoint to {final_checkpoint_path}")

    # Close TensorBoard writer
    writer.close()
    print(f"TensorBoard logs saved to {training_config.log_dir}")
    print("To view logs, run: tensorboard --logdir logs")


if __name__ == "__main__":
    main()
