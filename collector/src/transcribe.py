import os
import sys
import json
from pathlib import Path
import assemblyai as aai

# Get API key from environment
api_key = os.getenv("ASSEMBLYAI_API_KEY")
if not api_key:
    print("Error: Please set the ASSEMBLYAI_API_KEY environment variable.")
    sys.exit(1)

aai.settings.api_key = api_key


def save_json_next_to_file(original_file, data):
    p = Path(original_file)
    json_path = p.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved transcription to {json_path}")


def main():
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <audio_file>")
        sys.exit(1)
    audio_file = sys.argv[1]
    transcriber = aai.Transcriber(
        config=aai.TranscriptionConfig(
            language_code="nl",
            speaker_labels=True,
            punctuate=True,
            format_text=True,
            # From the Parakeet paper:
            # We should have both in the training set because it is awkward to write disfluencies in a prompt
            # How much is a bit unclear
            # TODO: However, this option is not supported for Dutch :(
            # disfluencies=True
        )
    )
    transcript = transcriber.transcribe(audio_file)
    save_json_next_to_file(audio_file, transcript.json_response)


if __name__ == "__main__":
    main()
