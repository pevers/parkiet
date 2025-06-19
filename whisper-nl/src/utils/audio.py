import subprocess
import json
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)

def get_audio_duration(audio_file: Path) -> Optional[float]:
    """Get audio file duration using ffprobe"""
    try:
        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            str(audio_file),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            info = json.loads(result.stdout)
            return float(info["format"]["duration"])
    except Exception as e:
        logger.error(f"Error getting duration for {audio_file}: {e}")
    return None

def extract_audio_chunk(
    input_audio: Path,
    output_audio: Path,
    start_time: float,
    end_time: float,
    sample_rate: int = 16000,
    channels: int = 1,
) -> bool:
    """Extract and convert audio chunk using ffmpeg"""
    try:
        duration = end_time - start_time
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_audio),
            "-ss",
            str(start_time),
            "-t",
            str(duration),
            "-ar",
            str(sample_rate),
            "-ac",
            str(channels),
            "-c:a",
            "pcm_s16le",
            str(output_audio),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0
    except Exception as e:
        logger.error(f"Error extracting audio chunk: {e}")
        return False 