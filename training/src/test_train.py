"""Script to test the training loop with minimal resources."""
import os
import json
import torch
from train import TrainingConfig, compute_loss
from dia.model import DiaModel
from dia.config import DiaConfig

def create_dummy_batch(batch_size: int, config: DiaConfig, device: str) -> dict:
    """Create a dummy batch for testing."""
    return {
        "text": torch.randint(
            0, config.model.src_vocab_size,
            (batch_size, config.data.text_length),
            device=device
        ),
        "audio": torch.randint(
            0, config.model.tgt_vocab_size - 3,  # -3 for special tokens
            (batch_size, config.data.audio_length, config.data.channels),
            device=device
        )
    }

def main():
    # Load test config
    with open("config.test.json") as f:
        config_dict = json.load(f)
    
    model_config = DiaConfig(config_dict)
    
    # Minimal training config
    training_config = TrainingConfig(
        batch_size=1,
        learning_rate=1e-4,
        num_epochs=1,
        warmup_steps=10,
        total_steps=100,
        gradient_accumulation_steps=1,
        fp16=True,
        checkpoint_dir="weights/test"
    )
    
    # Initialize model
    device = torch.device(training_config.device)
    print(f"Using device: {device}")
    
    print("Initializing model...")
    model = DiaModel(model_config)
    model.to(device)
    
    # Print model size
    param_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
    print(f"Model size: {param_size:.2f} MB")
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )
    
    # Initialize mixed precision training
    scaler = torch.cuda.amp.GradScaler() if training_config.fp16 else None
    
    print("Starting test training loop...")
    model.train()
    
    # Run 10 steps of training
    for step in range(10):
        # Create dummy batch
        batch = create_dummy_batch(training_config.batch_size, model_config, device)
        
        # Mixed precision training
        if training_config.fp16:
            with torch.cuda.amp.autocast():
                loss_dict = compute_loss(
                    model,
                    batch["text"],
                    batch["audio"],
                    training_config.text_dropout_prob,
                )
                loss = loss_dict["loss"]
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_dict = compute_loss(
                model,
                batch["text"],
                batch["audio"],
                training_config.text_dropout_prob,
            )
            loss = loss_dict["loss"]
            loss.backward()
            optimizer.step()
        
        optimizer.zero_grad()
        
        print(f"Step {step}, Loss: {loss.item():.4f}")
    
    print("Test training completed successfully!")

if __name__ == "__main__":
    main() 