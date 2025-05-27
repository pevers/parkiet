from pydantic import BaseModel
from typing import List, Optional, Dict, Any


class Chunk(BaseModel):
    speaker: str
    start: float  # milliseconds
    end: float  # milliseconds
    confidence: float
    text: str


class Segment(BaseModel):
    audio_url: str
    start: float  # milliseconds
    end: float  # milliseconds
    confidence: float
    text: str
    chunks: List[Chunk]
    emotion: Optional[str] = None


class ConversationPair(BaseModel):
    audio_url: str
    start: float  # milliseconds
    end: float  # milliseconds
    duration: float  # milliseconds
    speaker_a: str
    speaker_b: str
    text_a: str
    text_b: str
    confidence_a: float
    confidence_b: float
    emotion_a: Optional[str] = None
    emotion_b: Optional[str] = None
    gap_duration: float  # milliseconds between segments


class SingleSegment(BaseModel):
    audio_url: str
    start: float  # milliseconds
    end: float  # milliseconds
    duration: float  # milliseconds
    speaker: str
    text: str
    confidence: float
    emotion: Optional[str] = None


class Prompt(BaseModel):
    type: str  # "pair" or "single"
    data: Dict[str, Any]  # ConversationPair or SingleSegment data
