#!/usr/bin/env python3
"""
Test script for the audio preprocessing functionality.

This script demonstrates how to use the preprocess_audio module to:
1. Extract speaker events using pyannote
2. Create audio chunks with a 30-second sliding window
3. Skip the first and last 2 minutes (with clean silence cutoffs)
4. Store results in the data/chunks directory structure

Usage:
    python test_preprocess.py <audio_file> <output_name>

Example:
    python test_preprocess.py data/podcasts/example.mp3 example_episode
"""

import sys
from pathlib import Path

# Add src to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

from preprocess_audio import preprocess_audio  # type: ignore


def main():
    """Run a test of the audio preprocessing pipeline."""
    if len(sys.argv) != 3:
        print("Usage: python test_preprocess.py <audio_file> <output_name>")
        print(
            "Example: python test_preprocess.py data/podcasts/example.mp3 example_episode"
        )
        sys.exit(1)

    audio_file = sys.argv[1]
    output_name = sys.argv[2]

    print("=" * 60)
    print("Testing Audio Preprocessing Pipeline")
    print("=" * 60)
    print(f"Input audio: {audio_file}")
    print(f"Output name: {output_name}")
    print()

    try:
        # Run the preprocessing
        preprocess_audio(audio_file, output_name)

        print()
        print("=" * 60)
        print("Preprocessing completed successfully!")
        print("=" * 60)

        # Show the results
        output_dir = Path("data/chunks") / output_name
        if output_dir.exists():
            print(f"\nResults saved to: {output_dir}")
            print("\nGenerated files:")
            for file_path in sorted(output_dir.iterdir()):
                if file_path.is_file():
                    print(f"  - {file_path.name}")

    except Exception as e:
        print(f"Error during preprocessing: {e}")
        print("\nMake sure you have:")
        print("1. Installed pyannote.audio: pip install pyannote.audio")
        print(
            "2. Accepted the user conditions at: https://huggingface.co/pyannote/speaker-diarization"
        )
        print("3. Have ffmpeg installed for audio processing")
        sys.exit(1)


if __name__ == "__main__":
    main()
