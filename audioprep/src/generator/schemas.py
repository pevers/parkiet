from pydantic import BaseModel
from dataclasses import dataclass


@dataclass
class SpeakerEvent:
    """Represents a speaker event from pyannote diarization."""

    start: float  # seconds
    end: float  # seconds
    speaker: str


class AudioChunk(BaseModel):
    """Represents an audio chunk from preprocessing."""

    start: float  # milliseconds
    end: float  # milliseconds
    speaker_events: list[SpeakerEvent]
    file_path: str  # relative path to the chunk file


class ProcessedAudioData(BaseModel):
    """Represents all processed data for a single audio file."""

    source_file: str
    output_directory: str
    audio_duration_sec: float
    chunks: list[AudioChunk]
    processing_window: dict[str, float]  # start and end times used for processing
    success: bool
