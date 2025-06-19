import os
import torch
from pathlib import Path
from pyannote.audio import Pipeline
from audioprep.generator.schemas import SpeakerEvent

ACCESS_TOKEN = os.getenv("HF_TOKEN")


def extract_speaker_events(vocals_path: Path, device: str = "cuda") -> list[SpeakerEvent]:
    """
    Extract speaker events using pyannote speaker diarization

    Args:
        vocals_path: Path to the file

    Returns:
        List of SpeakerEvent objects
    """
    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", use_auth_token=ACCESS_TOKEN
    )
    diarization_pipeline.to(torch.device(device))
    diarization = diarization_pipeline(vocals_path.as_posix())

    events = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        events.append(
            SpeakerEvent(start=turn.start, end=turn.end, speaker=str(speaker))
        )

    # Sort by start time
    events.sort(key=lambda x: x.start)
    return events
