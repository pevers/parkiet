import argparse
import logging
import random
from pathlib import Path
import time
from dotenv import load_dotenv

load_dotenv()
from parkiet.database.audio_store import AudioStore
from parkiet.database.redis_connection import get_redis_connection

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)

def queue_audio_file(
    gcs_audio_path: str,
    queue_name: str = "audio_processing",
    window_size_sec: float = 30.0,
    skip_start_sec: float = 120.0,
    skip_end_sec: float = 180.0,
):
    """
    Queue a GCS audio file for processing.

    Args:
        gcs_audio_path: GCS path to the audio file
        queue_name: Redis queue name
        window_size_sec: Size of sliding window in seconds
        skip_start_sec: Time to skip from start
        skip_end_sec: Time to skip from end

    Returns:
        True if successfully queued, False otherwise
    """
    # Check if file has already been processed
    audio_store = AudioStore()
    if audio_store.is_audio_file_processed(gcs_audio_path):
        log.info(f"Audio file {gcs_audio_path} has already been processed, skipping")
        return False

    redis_client = get_redis_connection()

    job_data = {
        "audio_file_path": gcs_audio_path,
        "window_size_sec": window_size_sec,
        "skip_start_sec": skip_start_sec,
        "skip_end_sec": skip_end_sec,
        "timestamp": time.time(),
    }

    redis_client.push_job(queue_name, job_data)
    return True


def queue_from_file(
    file_path: str,
    queue_name: str = "audio_processing",
    window_size_sec: float = 30.0,
    skip_start_sec: float = 120.0,
    skip_end_sec: float = 180.0,
) -> int:
    """
    Queue GCS audio files from a file list for processing.

    Args:
        file_path: Path to file containing GCS MP3 paths (one per line)
        queue_name: Redis queue name
        window_size_sec: Size of sliding window in seconds
        skip_start_sec: Time to skip from start
        skip_end_sec: Time to skip from end

    Returns:
        Number of files queued
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    # Read GCS MP3 file paths from file
    with open(path, "r", encoding="utf-8") as f:
        gcs_mp3_paths = [line.strip() for line in f if line.strip()]

    if not gcs_mp3_paths:
        log.warning(f"No GCS MP3 paths found in: {path}")
        return 0

    log.info(f"Found {len(gcs_mp3_paths)} GCS MP3 paths in: {path}")

    # Random shuffle the list because we will probably not be able to process all files
    random.shuffle(gcs_mp3_paths)

    # Check which files have already been processed
    audio_store = AudioStore()
    processed_count = 0
    queued_count = 0

    for gcs_mp3_path in gcs_mp3_paths[0:10000]:
        if audio_store.is_audio_file_processed(gcs_mp3_path):
            log.info(f"Audio file {gcs_mp3_path} has already been processed, skipping")
            processed_count += 1
        else:
            success = queue_audio_file(
                gcs_mp3_path,
                queue_name,
                window_size_sec,
                skip_start_sec,
                skip_end_sec,
            )
            if success:
                queued_count += 1

    log.info(
        f"Queued {queued_count} out of {len(gcs_mp3_paths)} audio files ({processed_count} already processed)"
    )
    return queued_count


def process_input(
    input_path: str,
    queue_name: str = "audio_processing",
    window_size_sec: float = 30.0,
    skip_start_sec: float = 120.0,
    skip_end_sec: float = 180.0,
) -> int:
    """
    Process input which can be either a GCS path or a file containing GCS paths.

    Args:
        input_path: Either a GCS path or a file containing GCS paths
        queue_name: Redis queue name
        window_size_sec: Size of sliding window in seconds
        skip_start_sec: Time to skip from start
        skip_end_sec: Time to skip from end

    Returns:
        Number of files queued
    """
    # Check if input is a GCS path (starts with gs://)
    if input_path.startswith("gs://"):
        success = queue_audio_file(
            input_path,
            queue_name,
            window_size_sec,
            skip_start_sec,
            skip_end_sec,
        )
        if success:
            log.info(f"Successfully queued: {input_path}")
            return 1
        else:
            log.info(f"File {input_path} was already processed, not queued")
            return 0
    else:
        # Treat as a file containing GCS paths
        return queue_from_file(
            input_path,
            queue_name,
            window_size_sec,
            skip_start_sec,
            skip_end_sec,
        )


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Audio file queueing with Redis support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
The input can be either:
  - A GCS path (e.g., gs://bucket/path/to/audio/file.mp3)
  - A file containing GCS paths (one per line)

Examples:
  %(prog)s gs://bucket/path/to/audio/file.mp3
  %(prog)s bucket_mp3_files.txt
        """,
    )

    parser.add_argument(
        "input_path",
        help="GCS path to audio file or file containing GCS paths"
    )
    parser.add_argument(
        "--queue-name",
        "-q",
        default="audio_processing",
        help="Redis queue name (default: audio_processing)",
    )
    parser.add_argument(
        "--window-size",
        "-s",
        type=float,
        default=30.0,
        help="Size of sliding window in seconds (default: 30.0)",
    )
    parser.add_argument(
        "--skip-start",
        type=float,
        default=120.0,
        help="Time to skip from start in seconds (default: 120.0)",
    )
    parser.add_argument(
        "--skip-end",
        type=float,
        default=180.0,
        help="Time to skip from end in seconds (default: 180.0)",
    )

    args = parser.parse_args()

    count = process_input(
        args.input_path,
        args.queue_name,
        args.window_size,
        args.skip_start,
        args.skip_end,
    )
    
    if args.input_path.startswith("gs://"):
        # Single file case - count is already logged in process_input
        pass
    else:
        # File list case
        log.info(f"Queued {count} files")


if __name__ == "__main__":
    main()
