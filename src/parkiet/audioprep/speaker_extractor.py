import os
from pyannote.core import Segment
import torch
import numpy as np
from pathlib import Path
from pyannote.audio import Pipeline, Model
from parkiet.audioprep.schemas import SpeakerEvent
from pyannote.audio import Audio, Inference
import logging

ACCESS_TOKEN = os.getenv("HF_TOKEN")

logger = logging.getLogger(__name__)


class SpeakerExtractor:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", use_auth_token=ACCESS_TOKEN
        )
        self.diarization_pipeline.to(torch.device(device))

        # Initialize speaker embedding model
        self.embedding_model = Model.from_pretrained(
            "pyannote/embedding", use_auth_token=ACCESS_TOKEN
        )
        self.embedding_model.to(torch.device(device))

    def extract_speaker_events(
        self, vocals_path: Path
    ) -> tuple[list[SpeakerEvent], dict[str, np.ndarray]]:
        """
        Extract speaker events and embeddings per unique speaker

        Args:
            vocals_path: Path to the file

        Returns:
            Tuple of (events, speaker_embeddings)
            - events: List of SpeakerEvent objects
            - speaker_embeddings: Dict mapping speaker_id to embedding array
        """
        diarization = self.diarization_pipeline(vocals_path.as_posix())
        events = []
        speaker_segments = {}

        # First pass: collect events and group segments by speaker
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speaker_id = str(speaker)
            events.append(
                SpeakerEvent(start=turn.start, end=turn.end, speaker=speaker_id)
            )

            if speaker_id not in speaker_segments:
                speaker_segments[speaker_id] = []
            speaker_segments[speaker_id].append(turn)

        events.sort(key=lambda x: x.start)

        # Second pass: extract embeddings for each unique speaker
        speaker_embeddings = {}
        for speaker_id, segments in speaker_segments.items():
            # Use the longest segment for embedding extraction
            longest_segment = max(segments, key=lambda x: x.end - x.start)

            # Extract embedding from the segment
            embedding = self._extract_embedding_from_segment(
                vocals_path, longest_segment
            )

            if embedding is not None:
                speaker_embeddings[speaker_id] = embedding

        return events, speaker_embeddings

    def _extract_embedding_from_segment(self, audio_path: Path, segment):
        """
        Extract speaker embedding from a specific audio segment

        Args:
            audio_path: Path to the audio file
            segment: Pyannote segment object

        Returns:
            Speaker embedding as numpy array
        """
        inference = Inference(self.embedding_model, window="whole")
        embedding = inference.crop(audio_path, segment)
        return embedding
