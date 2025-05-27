from pathlib import Path
import json
import sys
import subprocess
import ulid
from typing import Dict
from schemas import Segment, ConversationPair, SingleSegment, Prompt


def get_primary_speaker(segment: Segment) -> str:
    """Get the primary speaker for a segment based on the chunks."""
    if not segment.chunks:
        return "unknown"

    # Count words per speaker
    speaker_counts: Dict[str, int] = {}
    for chunk in segment.chunks:
        speaker_counts[chunk.speaker] = speaker_counts.get(chunk.speaker, 0) + 1

    # Return the speaker with the most words
    return max(speaker_counts.items(), key=lambda x: x[1])[0]


def extract_audio_segment(
    original_audio_path: str,
    start_ms: float,
    end_ms: float,
    output_path: str,
    sample_rate: int = 16000,
) -> None:
    """Extract audio segment using ffmpeg."""
    start_sec = start_ms / 1000.0
    duration_sec = (end_ms - start_ms) / 1000.0

    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-i",
        original_audio_path,
        "-ss",
        str(start_sec),
        "-t",
        str(duration_sec),
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        output_path,
    ]

    subprocess.run(
        ffmpeg_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )


def generate_samples(
    chunks_folder: str,
    original_audio_path: str,
    output_folder: str,
    max_duration_ms: float = 30000,
    max_gap_ms: float = 2000,
    min_confidence: float = 0.5,
) -> None:
    """
    Generate conversation pairs and single segments from chunks.

    Args:
        chunks_folder: Path to folder containing chunks.json
        original_audio_path: Path to the original MP3 file
        output_folder: Output folder for audio files and prompts.json
        max_duration_ms: Maximum duration for pairs/segments in milliseconds
        max_gap_ms: Maximum gap between segments to consider them a pair
        min_confidence: Minimum confidence threshold for segments
    """
    chunks_path = Path(chunks_folder) / "chunks.json"
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    if not chunks_path.exists():
        raise FileNotFoundError(f"chunks.json not found in {chunks_folder}")

    if not Path(original_audio_path).exists():
        raise FileNotFoundError(f"Original audio file not found: {original_audio_path}")

    # Load segments
    with open(chunks_path, "r") as f:
        segments_data = json.load(f)

    segments = [Segment(**seg) for seg in segments_data]

    # Filter segments by confidence
    segments = [seg for seg in segments if seg.confidence >= min_confidence]

    prompts = []

    # Process segments to create pairs and singles
    i = 0
    while i < len(segments):
        current_segment = segments[i]
        current_duration = current_segment.end - current_segment.start
        current_speaker = get_primary_speaker(current_segment)

        # Skip if current segment is too long
        if current_duration > max_duration_ms:
            print(f"Skipping segment {i}: too long ({current_duration:.0f}ms)")
            i += 1
            continue

        # Try to create a pair with the next segment
        if i + 1 < len(segments):
            next_segment = segments[i + 1]
            next_speaker = get_primary_speaker(next_segment)

            # Calculate gap and total duration
            gap_duration = next_segment.start - current_segment.end
            total_duration = next_segment.end - current_segment.start

            # Check if we can create a pair
            can_pair = (
                total_duration <= max_duration_ms
                and gap_duration <= max_gap_ms
                and gap_duration >= 0  # Ensure no overlap
                and next_segment.confidence >= min_confidence
            )

            if can_pair:
                # Create conversation pair
                audio_ulid = str(ulid.new())
                audio_filename = f"{audio_ulid}.mp3"
                audio_out_path = output_path / audio_filename

                # Extract audio
                extract_audio_segment(
                    original_audio_path,
                    current_segment.start,
                    next_segment.end,
                    str(audio_out_path),
                )

                pair = ConversationPair(
                    audio_url=audio_filename,
                    start=current_segment.start,
                    end=next_segment.end,
                    duration=total_duration,
                    speaker_a=current_speaker,
                    speaker_b=next_speaker,
                    text_a=current_segment.text,
                    text_b=next_segment.text,
                    confidence_a=current_segment.confidence,
                    confidence_b=next_segment.confidence,
                    emotion_a=getattr(current_segment, "emotion", None),
                    emotion_b=getattr(next_segment, "emotion", None),
                    gap_duration=gap_duration,
                )

                prompts.append(Prompt(type="pair", data=pair.model_dump()))
                print(
                    f"Created pair: {current_speaker} -> {next_speaker} ({total_duration:.0f}ms)"
                )

                # Skip both segments
                i += 2
                continue

        # Create single segment if we couldn't pair or it's the last segment
        audio_ulid = str(ulid.new())
        audio_filename = f"{audio_ulid}.mp3"
        audio_out_path = output_path / audio_filename

        # Extract audio
        extract_audio_segment(
            original_audio_path,
            current_segment.start,
            current_segment.end,
            str(audio_out_path),
        )

        single = SingleSegment(
            audio_url=audio_filename,
            start=current_segment.start,
            end=current_segment.end,
            duration=current_duration,
            speaker=current_speaker,
            text=current_segment.text,
            confidence=current_segment.confidence,
            emotion=getattr(current_segment, "emotion", None),
        )

        prompts.append(Prompt(type="single", data=single.model_dump()))
        print(f"Created single: {current_speaker} ({current_duration:.0f}ms)")

        i += 1

    # Save prompts.json
    prompts_path = output_path / "prompts.json"
    with open(prompts_path, "w") as f:
        json.dump([prompt.model_dump() for prompt in prompts], f, indent=2)

    print(f"\nGenerated {len(prompts)} prompts:")
    pair_count = sum(1 for p in prompts if p.type == "pair")
    single_count = sum(1 for p in prompts if p.type == "single")
    print(f"  - {pair_count} conversation pairs")
    print(f"  - {single_count} single segments")
    print(f"Saved to {prompts_path}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(
            f"Usage: python {sys.argv[0]} <chunks_folder> <original_audio_path> <output_folder>"
        )
        print(
            "Example: python src/sample_generator.py data/chunks/podcast_name/ data/podcasts/podcast_name.mp3 output/samples/"
        )
        sys.exit(1)

    chunks_folder = sys.argv[1]
    original_audio_path = sys.argv[2]
    output_folder = sys.argv[3]

    generate_samples(chunks_folder, original_audio_path, output_folder)
