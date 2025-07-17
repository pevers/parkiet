from pydantic import BaseModel


class SpeakerEvent(BaseModel):
    """Represents a speaker event from pyannote diarization."""

    start: float  # seconds
    end: float  # seconds
    speaker: str


class AudioChunk(BaseModel):
    """Represents an audio chunk from preprocessing."""

    start: float  # milliseconds
    end: float  # milliseconds
    speaker_events: list[SpeakerEvent]
    file_path: str  # local file path to the chunk file
    gcs_file_path: str  # GCS path to the chunk file


class ProcessedAudioChunk(BaseModel):
    """Represents an audio chunk with transcription and speaker events."""

    audio_chunk: AudioChunk
    transcription: str
    transcription_clean: str


class ProcessedAudioFile(BaseModel):
    """Represents an audio file with processed chunks."""

    source_file: str
    output_directory: str
    gcs_audio_path: str
    audio_duration_sec: float
    chunks: list[ProcessedAudioChunk]
    success: bool
    processing_window: dict[str, float] = {"start": 0.0, "end": 0.0}
