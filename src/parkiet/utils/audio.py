import subprocess
from pathlib import Path
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
        "-c:a",
        "pcm_s16le",  # Use 16-bit PCM encoding for WAV
        output_path.as_posix(),
    ]

    try:
        subprocess.run(
            ffmpeg_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        log.info(f"Successfully extracted audio segment to {output_path}")
    except subprocess.CalledProcessError as e:
        log.error(f"FFmpeg failed to extract audio segment {output_path}: {e}")
        raise
    except Exception as e:
        log.error(f"Unexpected error extracting audio segment {output_path}: {e}")
        raise


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
    if not chunk_tasks:
        return
        
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks and collect futures with their corresponding tasks
        future_to_task = {
            executor.submit(extract_audio_segment, *task): task 
            for task in chunk_tasks
        }
        
        # Wait for all futures to complete and handle individual failures
        failed_count = 0
        for future in concurrent.futures.as_completed(future_to_task):
            task = future_to_task[future]
            try:
                future.result()  # This will raise any exception that occurred
            except Exception as e:
                # Log the error but don't stop processing other chunks
                failed_count += 1
                log.error(f"Failed to extract audio segment {task[3]}: {e}")
        
        successful_count = len(chunk_tasks) - failed_count
        log.info(f"Parallel audio extraction completed: {successful_count} successful, {failed_count} failed out of {len(chunk_tasks)} total")


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


def convert_to_wav(
    input_path: Path,
    output_path: Path,
    sample_rate: int = 16000,
    channels: int = 1,
) -> None:
    """
    Convert audio file to WAV format using ffmpeg.
    
    Args:
        input_path: Path to the input audio file
        output_path: Path for the output WAV file
        sample_rate: Target sample rate in Hz (default: 16000)
        channels: Number of audio channels (default: 1 for mono)
    """
    log.info(f"Converting {input_path} to WAV format at {output_path}")
    
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",  # Overwrite output file if it exists
        "-i", input_path.as_posix(),
        "-ar", str(sample_rate),  # Set sample rate
        "-ac", str(channels),     # Set number of channels
        "-c:a", "pcm_s16le",      # Use 16-bit PCM encoding
        output_path.as_posix(),
    ]
    
    try:
        subprocess.run(
            ffmpeg_cmd, 
            check=True, 
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.DEVNULL
        )
        log.info(f"Successfully converted {input_path} to {output_path}")
    except subprocess.CalledProcessError as e:
        log.error(f"Failed to convert {input_path} to WAV: {e}")
        raise


def find_natural_break_after_time(
    speaker_events: list[SpeakerEvent], target_time_sec: float
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
