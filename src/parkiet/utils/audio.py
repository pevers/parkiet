import subprocess
from pathlib import Path
from typing import List
import concurrent.futures
import logging
from parkiet.audioprep.schemas import SpeakerEvent

log = logging.getLogger(__name__)


def validate_audio_file(audio_path: Path) -> bool:
    """
    Validate audio file integrity using ffprobe.
    
    Args:
        audio_path: Path to the audio file to validate
        
    Returns:
        True if file is valid, False if corrupted or invalid
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=codec_type,duration",
        "-of", "csv=p=0",
        audio_path.as_posix(),
    ]
    
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=10  # Short timeout for validation
        )
        
        if result.returncode != 0:
            log.error(f"Audio file validation failed for {audio_path}: {result.stderr}")
            return False
            
        # Check if we got valid output
        output = result.stdout.strip()
        if not output or "audio" not in output:
            log.error(f"Invalid audio stream in {audio_path}")
            return False
            
        log.info(f"Audio file {audio_path} validated successfully")
        return True
        
    except subprocess.TimeoutExpired:
        log.error(f"Timeout validating audio file {audio_path} - likely corrupted")
        return False
    except Exception as e:
        log.error(f"Error validating audio file {audio_path}: {e}")
        return False


def get_audio_duration(audio_path: Path) -> float:
    """Get audio duration in seconds using ffprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-show_entries",
        "format=duration",
        "-of",
        "csv=p=0",
        audio_path.as_posix(),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


def extract_audio_segment(
    original_audio_path: Path,
    start_sec: float,
    end_sec: float,
    output_path: Path,
    sample_rate: int = 16000,
) -> None:
    """Extract audio segment using ffmpeg."""
    log.info(
        f"Extracting audio segment from {original_audio_path} from {start_sec}s to {end_sec}s to {output_path}"
    )
    duration_sec = end_sec - start_sec

    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-i",
        original_audio_path.as_posix(),
        "-ss",
        str(start_sec),
        "-t",
        str(duration_sec),
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        output_path.as_posix(),
    ]

    subprocess.run(
        ffmpeg_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )


def extract_audio_segments_parallel(
    chunk_tasks: list[tuple[Path, float, float, Path]],
    max_workers: int = 4,
) -> None:
    """
    Extract multiple audio segments in parallel using ThreadPoolExecutor.

    Args:
        chunk_tasks: List of tuples (original_audio_path, start_sec, end_sec, output_path)
        max_workers: Maximum number of parallel workers (default: 4)
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks and wait for completion
        list(executor.map(lambda task: extract_audio_segment(*task), chunk_tasks))


def find_audio_files(source_folder: Path) -> list[Path]:
    """
    Find all audio files in the source folder recursively.

    Args:
        source_folder: Path to the source folder

    Returns:
        List of audio file paths
    """
    # Supported audio file extensions
    AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".wma"}

    audio_files = []
    for file_path in source_folder.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in AUDIO_EXTENSIONS:
            audio_files.append(file_path)

    return sorted(audio_files)


def find_natural_break_after_time(
    speaker_events: List[SpeakerEvent], target_time_sec: float
) -> float:
    """
    Find a natural break (end of speaker segment) after target_time_sec.

    Args:
        speaker_events: List of speaker events
        target_time_sec: Time in seconds after which to look for a break

    Returns:
        Time in seconds where a natural break occurs after target_time_sec
    """
    for event in speaker_events:
        if event.end >= target_time_sec:
            return event.end

    # If no event found after target time, return target time
    return target_time_sec
