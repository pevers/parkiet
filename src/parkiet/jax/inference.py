from parkiet.jax.model import Dia
from parkiet.torch.dataset import create_dataset
from parkiet.dia.config import DiaConfig
from pathlib import Path
import numpy as np
import random

# Load model
checkpoint_path = (Path("weights") / "checkpoint_2000").resolve()
config = DiaConfig.from_json("config.test.json")
model = Dia.from_local(
    config_path="config.test.json",
    checkpoint_path=checkpoint_path.as_posix(),
    compute_dtype="float32",
    param_dtype="float32",
)

# Load dataset
dataset = create_dataset(
    parquet_path="chunks_dataset.test.parquet",
    config=config,
)

print(f"Dataset loaded with {len(dataset)} samples")

# Get a random sample from the dataset
sample_idx = random.randint(0, len(dataset) - 1)
sample = dataset[sample_idx]
sample_info = dataset.get_sample_info(sample_idx)

print(f"Selected sample {sample_idx}")
print(f"Source: {sample_info['source_file']}")
print(f"Duration: {sample_info['duration_ms']}ms")
print(f"Text: {sample_info['transcription']}")

# Get the text prompt and audio
text_tokens = sample["text"]
full_audio = sample["audio_target"]  # Use target which has EOS

print(f"Text tokens shape: {text_tokens.shape}")
print(f"Audio shape: {full_audio.shape}")

# Split audio in half
audio_length = full_audio.shape[0]
half_length = audio_length // 2

# First half of audio (input for prefilling)
audio_prefix = full_audio[:half_length]
print(f"Audio prefix shape: {audio_prefix.shape}")

# Test: Generate continuation from prefilled audio + text
text_str = sample_info['transcription']

print(f"\nGenerating from text: {text_str}")
print(f"With audio prefix of length: {half_length}")

# Use the audio prefix as audio prompt for generation
# The audio_prefix needs to be converted back from delayed format
# For now, just use it as is - the model should handle the delay pattern internally
output = model.generate(
    text_str,
    audio_prompt=audio_prefix,
    verbose=True,
    cfg_scale=0.0,
    temperature=0.0,
    cfg_filter_top_k=1,
    max_tokens=300,
)

model.save_audio("prefill_test.mp3", output)
print("Saved generated audio as 'prefill_test.mp3'")

# Also save the original full audio for comparison
original_audio_tokens = full_audio
original_output = model.decode_audio_tokens(original_audio_tokens[None, :, :])
model.save_audio("original_full.mp3", original_output[0])
print("Saved original audio as 'original_full.mp3'")

# Save just the prefix audio for comparison
prefix_output = model.decode_audio_tokens(audio_prefix[None, :, :])
model.save_audio("prefix_only.mp3", prefix_output[0])
print("Saved prefix audio as 'prefix_only.mp3'")
