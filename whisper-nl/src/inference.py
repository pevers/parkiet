"""
Simple inference script for fine-tuned Whisper model
"""

import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import soundfile as sf
import librosa
import argparse
import os
from train_configurable import SPECIAL_TOKENS

def load_model(checkpoint_path):
    """Load the fine-tuned Whisper model and processor"""
    print(f"Loading model from {checkpoint_path}...")

    # Load processor and model
    # model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-large-v3", low_cpu_mem_usage=True, use_safetensors=True)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        checkpoint_path, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.generation_config.suppress_tokens = []
    processor = AutoProcessor.from_pretrained(
        checkpoint_path
    )

    # Move to GPU if available
    #device = torch.device("cpu")  # For debugging
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    additional_special_tokens = processor.tokenizer.additional_special_tokens
    print("Additional special tokens:", additional_special_tokens)

    print(f"Model loaded on {device}")
    return processor, model, device


def transcribe_audio(audio_path, processor, model, device):
    """Transcribe a single audio file"""
    print(f"Transcribing: {audio_path}")

    # Load audio
    audio, sr = sf.read(audio_path)
    print(f"Original sampling rate: {sr} Hz")

    # Ensure audio is mono
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    # Convert to 16 kHz if needed
    target_sr = 16000
    if sr != target_sr:
        print(f"Resampling from {sr} Hz to {target_sr} Hz...")
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
        print(f"Resampled to: {sr} Hz")
    else:
        print(f"Audio already at target sampling rate: {sr} Hz")

    # Process audio
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
    input_features = inputs.input_features.to(device)

    # Generate transcription
    with torch.no_grad():
        predicted_ids = model.generate(input_features)

    # Decode transcription
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=False)

    return transcription[0]


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio using fi  ne-tuned Whisper model"
    )
    parser.add_argument(
        "--audio_file",
        default="../collector/test/nl_laughter.mp3",
        help="Path to audio file to transcribe",
    )
    parser.add_argument(
        "--checkpoint",
        default="./whisper-small-dutch-cgn/checkpoint-1000",
        help="Path to model checkpoint (default: ./whisper-small-dutch-cgn/checkpoint-1000)",
    )

    args = parser.parse_args()

    # Check if files exist
    if not os.path.exists(args.audio_file):
        print(f"Error: Audio file {args.audio_file} not found")
        return

    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint directory {args.checkpoint} not found")
        return

    # Load model
    processor, model, device = load_model(args.checkpoint)

    # Transcribe audio
    transcription = transcribe_audio(args.audio_file, processor, model, device)

    print(f"\nTranscription:")
    print(f"'{transcription}'")


if __name__ == "__main__":
    main()
