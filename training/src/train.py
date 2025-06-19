import os
import json
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from dia.model import Dia, DiaModel
from dia.config import DiaConfig
from dataset import AudioTextDataset  # You'll need to implement this

class TrainingConfig:
    def __init__(self, **kwargs):
        self.batch_size: int = kwargs.get('batch_size', 256)
        self.learning_rate: float = kwargs.get('learning_rate', 2e-4)
        self.num_epochs: int = kwargs.get('num_epochs', 100)
        self.warmup_steps: int = kwargs.get('warmup_steps', 2000)
        self.total_steps: int = kwargs.get('total_steps', 110_000)
        self.text_dropout_prob: float = kwargs.get('text_dropout_prob', 0.15)
        self.gradient_accumulation_steps: int = kwargs.get('gradient_accumulation_steps', 1)
        self.fp16: bool = kwargs.get('fp16', True)
        self.checkpoint_dir: str = kwargs.get('checkpoint_dir', 'weights')
        self.checkpoint_every: int = kwargs.get('checkpoint_every', 1000)
        self.device: str = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

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
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + torch.cos(torch.pi * float(num_cycles) * 2.0 * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

def compute_loss(
    model: DiaModel,
    text_tokens: torch.Tensor,  # [B, T]
    audio_tokens: torch.Tensor,  # [B, S, C]
    text_dropout_prob: float = 0.15,
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
    batch_size = text_tokens.shape[0]
    device = text_tokens.device
    
    # For CFG training, randomly drop text condition for some samples
    # TODO: Make sure we are not doing the same in the data loader
    text_mask = torch.rand(batch_size, device=device) >= text_dropout_prob
    text_tokens_masked = text_tokens * text_mask.unsqueeze(-1)
    
    # Get encoder outputs
    encoder_outputs = model.encoder(text_tokens_masked)
    
    # Prepare decoder inputs and targets
    # Input sequence: [BOS, x1, x2, ..., xn]
    # Target sequence: [x1, x2, ..., xn, EOS]
    bos_token = model.config.data.audio_bos_value
    eos_token = model.config.data.audio_eos_value
    pad_token = model.config.data.audio_pad_value
    
    # Create input sequence starting with BOS
    bos_tokens = torch.full(
        (batch_size, 1, audio_tokens.shape[-1]), 
        bos_token, 
        device=device
    )
    decoder_input = torch.cat([bos_tokens, audio_tokens[:, :-1]], dim=1)
    
    # Create target sequence ending with EOS
    eos_tokens = torch.full(
        (batch_size, 1, audio_tokens.shape[-1]), 
        eos_token, 
        device=device
    )
    decoder_target = torch.cat([audio_tokens[:, 1:], eos_tokens], dim=1)
    
    # Get decoder outputs
    decoder_outputs = model.decoder(
        decoder_input,
        encoder_outputs=encoder_outputs,
    )
    
    # Calculate loss - only ignore padding tokens
    loss = nn.CrossEntropyLoss(ignore_index=pad_token)(
        decoder_outputs.view(-1, model.config.model.tgt_vocab_size),
        decoder_target.view(-1)
    )
    
    return {
        "loss": loss,
        "text_condition_ratio": text_mask.float().mean(),
    }

def train_epoch(
    model: DiaModel,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    scaler: Optional[GradScaler],
    config: TrainingConfig,
    epoch: int,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_steps = 0
    
    for batch_idx, batch in enumerate(train_loader):
        text_tokens = batch["text"].to(config.device)
        audio_tokens = batch["audio"].to(config.device)
        
        # Mixed precision training
        if config.fp16 and scaler is not None:
            with autocast():
                loss_dict = compute_loss(
                    model,
                    text_tokens,
                    audio_tokens,
                    config.text_dropout_prob,
                )
                loss = loss_dict["loss"] / config.gradient_accumulation_steps
                
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step()
        
        else:
            loss_dict = compute_loss(
                model,
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
            print(f"Epoch {epoch}, Step {batch_idx}, Loss: {loss.item():.4f}, LR: {lr:.6f}")
    
    return {
        "loss": total_loss / total_steps,
    }

def main():
    # Load config
    with open("config.json") as f:
        config_dict = json.load(f)
    
    model_config = DiaConfig(config_dict)
    training_config = TrainingConfig()
    
    # Initialize model
    device = torch.device(training_config.device)
    model = DiaModel(model_config)
    model.to(device)
    
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
    
    # Initialize mixed precision training
    scaler = GradScaler() if training_config.fp16 else None
    
    # Initialize data loader
    # Note: You need to implement the AudioTextDataset class
    train_dataset = AudioTextDataset(
        data_dir="path/to/data",
        config=model_config,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    
    # Training loop
    for epoch in range(training_config.num_epochs):
        metrics = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            config=training_config,
            epoch=epoch,
        )
        
        print(f"Epoch {epoch} completed. Average loss: {metrics['loss']:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % training_config.checkpoint_every == 0:
            os.makedirs(training_config.checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(
                training_config.checkpoint_dir,
                f"dia-v0_1-epoch{epoch+1}.pth"
            )
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

if __name__ == "__main__":
    main()
