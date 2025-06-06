#!/usr/bin/env python3
"""
Script to print default suppress tokens from Whisper tokenizer
"""

from transformers import WhisperProcessor, WhisperForConditionalGeneration
import argparse


def print_suppress_tokens(model_name="openai/whisper-small", language="dutch"):
    """Print the default suppress tokens from Whisper model"""
    print(f"Loading {model_name} with language={language}...")

    # Load processor and model
    processor = WhisperProcessor.from_pretrained(
        model_name, language=language, task="transcribe"
    )
    model = WhisperForConditionalGeneration.from_pretrained(model_name)

    print(f"\nModel: {model_name}")
    print(f"Language: {language}")
    print(f"Task: transcribe")
    print("=" * 80)

    # Get suppress tokens from generation config
    suppress_tokens = model.generation_config.suppress_tokens

    print(f"\nFound {len(suppress_tokens)} suppress tokens:")
    print(f"Suppress token IDs: {suppress_tokens}")
    print("\n" + "=" * 80)
    print("TOKEN DETAILS:")
    print("=" * 80)

    # Group tokens by ranges for better readability
    token_groups = {
        "Low IDs (0-100)": [],
        "Medium IDs (100-1000)": [],
        "High IDs (1000-10000)": [],
        "Very High IDs (10000+)": [],
    }

    # Decode each suppress token
    for token_id in suppress_tokens:
        try:
            # Decode the token
            decoded = processor.tokenizer.decode([token_id])

            # Group by token ID ranges
            if token_id < 100:
                token_groups["Low IDs (0-100)"].append((token_id, decoded))
            elif token_id < 1000:
                token_groups["Medium IDs (100-1000)"].append((token_id, decoded))
            elif token_id < 10000:
                token_groups["High IDs (1000-10000)"].append((token_id, decoded))
            else:
                token_groups["Very High IDs (10000+)"].append((token_id, decoded))

        except Exception as e:
            print(f"Error decoding token {token_id}: {e}")

    # Print grouped tokens
    for group_name, tokens in token_groups.items():
        if tokens:
            print(f"\n{group_name}:")
            print("-" * 40)
            for token_id, decoded in sorted(tokens):
                # Clean up the decoded string for display
                if decoded.strip():
                    display_text = repr(decoded)  # Use repr to show special characters
                else:
                    display_text = "<empty/whitespace>"
                print(f"  {token_id:5d}: {display_text}")

    # Print some additional info
    print("\n" + "=" * 80)
    print("ADDITIONAL INFO:")
    print("=" * 80)

    # Check for common patterns
    non_speech_events = []
    fillers = []
    punctuation = []

    for token_id in suppress_tokens:
        try:
            decoded = processor.tokenizer.decode([token_id]).lower()

            # Look for common patterns (this is approximate)
            if any(
                word in decoded
                for word in ["laugh", "applause", "music", "noise", "cough"]
            ):
                non_speech_events.append((token_id, decoded))
            elif any(word in decoded for word in ["um", "uh", "er", "ah"]):
                fillers.append((token_id, decoded))
            elif len(decoded.strip()) == 1 and not decoded.isalnum():
                punctuation.append((token_id, decoded))

        except:
            continue

    if non_speech_events:
        print(f"\nNon-speech events found: {len(non_speech_events)}")
        for token_id, decoded in non_speech_events:
            print(f"  {token_id}: {repr(decoded)}")

    if fillers:
        print(f"\nFiller words found: {len(fillers)}")
        for token_id, decoded in fillers:
            print(f"  {token_id}: {repr(decoded)}")

    if punctuation:
        print(f"\nPunctuation/symbols found: {len(punctuation)}")
        for token_id, decoded in punctuation:
            print(f"  {token_id}: {repr(decoded)}")

    # Show some special tokens for context
    print(f"\nSPECIAL TOKENS FOR REFERENCE:")
    print("-" * 40)
    print(f"BOS (beginning of sequence): {model.generation_config.bos_token_id}")
    print(f"EOS (end of sequence): {model.generation_config.eos_token_id}")
    print(f"PAD (padding): {model.generation_config.pad_token_id}")
    print(f"Decoder start: {model.generation_config.decoder_start_token_id}")

    # Language and task tokens
    if hasattr(model.generation_config, "lang_to_id"):
        dutch_token_id = model.generation_config.lang_to_id.get("<|nl|>")
        if dutch_token_id:
            print(f"Dutch language token: {dutch_token_id}")

    if hasattr(model.generation_config, "task_to_id"):
        transcribe_token_id = model.generation_config.task_to_id.get("transcribe")
        if transcribe_token_id:
            print(f"Transcribe task token: {transcribe_token_id}")


def main():
    parser = argparse.ArgumentParser(description="Print Whisper suppress tokens")
    parser.add_argument(
        "--model",
        default="openai/whisper-small",
        help="Whisper model to analyze (default: openai/whisper-small)",
    )
    parser.add_argument(
        "--language", default="dutch", help="Language setting (default: dutch)"
    )

    args = parser.parse_args()

    print_suppress_tokens(args.model, args.language)


if __name__ == "__main__":
    main()
