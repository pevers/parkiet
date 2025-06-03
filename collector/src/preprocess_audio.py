import json
import sys
import subprocess
import ulid
import librosa
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from pydantic import BaseModel
from pyannote.audio import Pipeline
from pyannote.core import Segment as PyannoteSegment
import os

ACCESS_TOKEN = os.getenv("HF_TOKEN")


class AudioChunk(BaseModel):
    """Represents an audio chunk from preprocessing."""

    speaker: str
    start: float  # milliseconds
    end: float  # milliseconds


@dataclass
class SpeakerEvent:
    """Represents a speaker event from pyannote diarization."""

    start: float  # seconds
    end: float  # seconds
    speaker: str

    def to_milliseconds(self) -> "SpeakerEvent":
        """Convert to milliseconds."""
        return SpeakerEvent(
            start=self.start * 1000, end=self.end * 1000, speaker=self.speaker
        )


def find_silence_after_time(
    audio_path: str, target_time_sec: float, min_silence_duration: float = 0.5
) -> float:
    """
    Find the first silence period after target_time_sec.

    Args:
        audio_path: Path to audio file
        target_time_sec: Time in seconds after which to look for silence
        min_silence_duration: Minimum duration of silence to consider (seconds)

    Returns:
        Time in seconds where silence starts after target_time_sec
    """
    # Load audio
    y, sr = librosa.load(audio_path, sr=16000)

    # Convert target time to samples
    target_sample = int(target_time_sec * sr)

    # If target time is beyond audio, return audio length
    if target_sample >= len(y):
        return len(y) / sr

    # Calculate RMS energy with a small hop length for better precision
    hop_length = 512
    frame_length = 2048
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

    # Convert to time
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

    # Find frames after target time
    start_frame = np.searchsorted(times, target_time_sec)

    # Define silence threshold (adjust based on your audio characteristics)
    silence_threshold = np.percentile(rms, 5)  # Use 5th percentile as silence threshold

    # Find consecutive silent frames
    min_silence_frames = int(min_silence_duration * sr / hop_length)

    for i in range(start_frame, len(rms) - min_silence_frames):
        if all(rms[i : i + min_silence_frames] < silence_threshold):
            return times[i]

    # If no silence found, return target time
    return target_time_sec


def extract_speaker_events(audio_path: str) -> List[SpeakerEvent]:
    """
    Extract speaker events using pyannote speaker diarization.

    Args:
        audio_path: Path to the audio file

    Returns:
        List of SpeakerEvent objects
    """
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization@2.1", use_auth_token=ACCESS_TOKEN
    )

    # Apply diarization
    diarization = pipeline(audio_path)

    events = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        events.append(
            SpeakerEvent(start=turn.start, end=turn.end, speaker=str(speaker))
        )

    # Sort by start time
    events.sort(key=lambda x: x.start)
    return events


def get_audio_duration(audio_path: str) -> float:
    """Get audio duration in seconds."""
    y, sr = librosa.load(audio_path, sr=None)
    return len(y) / sr


def extract_audio_segment(
    original_audio_path: str,
    start_sec: float,
    end_sec: float,
    output_path: str,
    sample_rate: int = 16000,
) -> None:
    """Extract audio segment using ffmpeg."""
    duration_sec = end_sec - start_sec

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


def create_chunks(
    speaker_events: List[SpeakerEvent],
    original_audio_path: str,
    output_dir: Path,
    window_size_sec: float = 30.0,
    skip_start_sec: float = 120.0,  # 2 minutes
    skip_end_sec: float = 180.0,  # 3 minutes
) -> List[AudioChunk]:
    """
    Create audio chunks from speaker events using a sliding window approach.

    Args:
        speaker_events: List of speaker events in seconds
        original_audio_path: Path to original audio file
        output_dir: Output directory for audio chunks
        window_size_sec: Size of sliding window in seconds
        skip_start_sec: Time to skip from start (will find silence after this)
        skip_end_sec: Time to skip from end (will find silence before this)

    Returns:
        List of AudioChunk objects
    """
    audio_duration = get_audio_duration(original_audio_path)

    # Find actual start and end times based on silence detection
    actual_start = find_silence_after_time(original_audio_path, skip_start_sec)
    actual_end = find_silence_after_time(
        original_audio_path, audio_duration - skip_end_sec
    )

    print(f"Audio duration: {audio_duration:.1f}s")
    print(f"Processing from {actual_start:.1f}s to {actual_end:.1f}s")

    # Filter events to the processing window
    events_in_window = []
    for event in speaker_events:
        # Only include events that overlap with our processing window
        if event.end > actual_start and event.start < actual_end:
            # Clip the event to our window
            clipped_start = max(event.start, actual_start)
            clipped_end = min(event.end, actual_end)
            events_in_window.append(
                SpeakerEvent(
                    start=clipped_start, end=clipped_end, speaker=event.speaker
                )
            )

    chunks = []
    current_time = actual_start

    while current_time < actual_end:
        window_end = min(current_time + window_size_sec, actual_end)

        # Find events in this window
        window_events = []
        for event in events_in_window:
            if event.start < window_end and event.end > current_time:
                # Clip event to window
                clipped_start = max(event.start, current_time)
                clipped_end = min(event.end, window_end)
                window_events.append(
                    SpeakerEvent(
                        start=clipped_start, end=clipped_end, speaker=event.speaker
                    )
                )

        if window_events:
            # Create chunk
            chunk_id = str(ulid.new())
            chunk_filename = f"{chunk_id}.mp3"
            chunk_path = output_dir / chunk_filename

            # Extract audio
            extract_audio_segment(
                original_audio_path, current_time, window_end, str(chunk_path)
            )

            # Find primary speaker (speaker with most time)
            speaker_durations: Dict[str, float] = {}
            for event in window_events:
                duration = event.end - event.start
                speaker_durations[event.speaker] = (
                    speaker_durations.get(event.speaker, 0) + duration
                )

            primary_speaker = (
                max(speaker_durations.items(), key=lambda x: x[1])[0]
                if speaker_durations
                else "unknown"
            )

            chunk = AudioChunk(
                speaker=primary_speaker,
                start=current_time * 1000,
                end=window_end * 1000,
            )

            chunks.append(chunk)
            print(
                f"Created chunk {len(chunks)}: {current_time:.1f}s-{window_end:.1f}s, speaker: {primary_speaker}"
            )

        # Move window forward (could be overlapping or non-overlapping based on requirements)
        current_time += window_size_sec  # Non-overlapping windows

    return chunks


def preprocess_audio(audio_path: str, output_name: str) -> None:
    """
    Main function to preprocess audio file.

    Args:
        audio_path: Path to the input audio file
        output_name: Name for the output directory (e.g., "podcast_episode_1")
    """
    audio_file_path = Path(audio_path)
    if not audio_file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_file_path}")

    # Create output directory
    output_dir = Path("data/chunks") / output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing audio file: {audio_file_path}")
    print(f"Output directory: {output_dir}")

    # Extract speaker events
    print("Extracting speaker events with pyannote...")
    speaker_events = extract_speaker_events(str(audio_file_path))

    print(f"Found {len(speaker_events)} speaker events")

    # Save speaker events as JSON
    events_json_path = output_dir / "speaker_events.json"
    with open(events_json_path, "w") as f:
        json.dump(
            [
                {"start": event.start, "end": event.end, "speaker": event.speaker}
                for event in speaker_events
            ],
            f,
            indent=2,
        )

    print(f"Saved speaker events to: {events_json_path}")

    # Create chunks
    print("Creating audio chunks...")
    chunks = create_chunks(speaker_events, str(audio_file_path), output_dir)

    # Save chunks as JSON
    chunks_json_path = output_dir / "chunks.json"
    with open(chunks_json_path, "w") as f:
        json.dump([chunk.model_dump() for chunk in chunks], f, indent=2)

    print(f"Created {len(chunks)} chunks")
    print(f"Saved chunks to: {chunks_json_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} <audio_path> <output_name>")
        print(
            "Example: python src/preprocess_audio.py data/podcasts/episode1.mp3 episode1"
        )
        sys.exit(1)

    audio_path = sys.argv[1]
    output_name = sys.argv[2]

    preprocess_audio(audio_path, output_name)
