import argparse
import logging
from pathlib import Path
import time
import torch
import shutil
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
from parkiet.audioprep.schemas import (
    AudioChunk,
    ProcessedAudioFile,
    ProcessedAudioChunk,
    SpeakerEvent,
)
from parkiet.utils.audio import (
    get_audio_duration,
    extract_audio_segment,
    find_audio_files,
    find_natural_break_after_time,
)
from parkiet.audioprep.speaker_extractor import SpeakerExtractor
from parkiet.audioprep.transcriber import Transcriber
from parkiet.database.audio_store import AudioStore
from parkiet.database.redis_connection import get_redis_connection
from parkiet.storage.gcs_client import get_gcs_client
from ulid import ULID

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)


class ChunkerWorker:
    """Worker class for processing audio files from Redis queue."""

    def __init__(
        self,
        queue_name: str = "audio_processing",
        whisper_checkpoint_path: str = "pevers/whisperd-nl",
        temp_dir: str = "/tmp/parkiet_chunks",
    ):
        """
        Initialize the chunker worker.

        Args:
            queue_name: Redis queue name to listen to
            whisper_checkpoint_path: Path to Whisper checkpoint
            temp_dir: Temporary directory for processing chunks
        """
        self.queue_name = queue_name
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        # Initialize services
        self.redis_client = get_redis_connection()
        self.gcs_client = get_gcs_client()
        self.audio_store = AudioStore()

        # Initialize AI models
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.speaker_extractor = SpeakerExtractor(device.type)
        self.transcriber = Transcriber(whisper_checkpoint_path, device)

    def process_job(self, job_data: dict) -> bool:
        """
        Process a single job from the queue.

        Args:
            job_data: Job data containing audio file path and processing parameters

        Returns:
            True if successful, False otherwise
        """
        try:
            # Extract job parameters
            audio_file_path = Path(job_data["audio_file_path"])
            window_size_sec = job_data.get("window_size_sec", 30.0)
            skip_start_sec = job_data.get("skip_start_sec", 120.0)
            skip_end_sec = job_data.get("skip_end_sec", 180.0)

            log.info(f"Processing job: {audio_file_path}")

            # Check if file has already been processed
            if self.audio_store.is_audio_file_processed(str(audio_file_path)):
                log.info(
                    f"Audio file {audio_file_path} has already been processed, skipping"
                )
                return True

            # Create temporary output directory for this job
            job_id = str(ULID())
            job_output_dir = self.temp_dir / job_id
            job_output_dir.mkdir(parents=True, exist_ok=True)

            try:
                # Process the audio file
                processed_file = self.process_single_audio_file(
                    audio_file_path,
                    job_output_dir,
                    window_size_sec,
                    skip_start_sec,
                    skip_end_sec,
                )

                if processed_file.success:
                    upload_success = self.upload_chunks_to_gcs(
                        processed_file, job_output_dir
                    )

                    if upload_success:
                        log.info(
                            f"Successfully processed and uploaded chunks for {audio_file_path}"
                        )
                        return True
                    else:
                        log.error(f"Failed to upload chunks for {audio_file_path}")
                        return False
                else:
                    log.error(f"Failed to process audio file {audio_file_path}")
                    return False

            finally:
                # Clean up temporary directory
                self.cleanup_temp_dir(job_output_dir)

        except Exception as e:
            log.error(f"Error processing job: {e}")
            return False

    def process_single_audio_file(
        self,
        audio_file_path: Path,
        output_dir: Path,
        window_size_sec: float = 30.0,
        skip_start_sec: float = 120.0,
        skip_end_sec: float = 180.0,
    ) -> ProcessedAudioFile:
        """
        Process a single audio file.

        Args:
            audio_file_path: Path to the audio file
            output_dir: Output directory for chunks
            window_size_sec: Size of sliding window in seconds
            skip_start_sec: Time to skip from start
            skip_end_sec: Time to skip from end

        Returns:
            ProcessedAudioFile object
        """
        start_time = time.time()

        try:
            log.info(f"Processing: {audio_file_path.name}")
            log.info(f"Output directory: {output_dir}")
            log.info("Extracting speaker events with pyannote...")
            speaker_events, speaker_embeddings = (
                self.speaker_extractor.extract_speaker_events(audio_file_path)
            )
            log.info(
                f"Found {len(speaker_events)} speaker events for {len(speaker_embeddings)} unique speakers"
            )

            if not speaker_events:
                log.warning("No speaker events found, skipping chunk creation")
                return ProcessedAudioFile(
                    source_file=audio_file_path.as_posix(),
                    output_directory=output_dir.as_posix(),
                    audio_duration_sec=get_audio_duration(audio_file_path),
                    chunks=[],
                    processing_window={"start": 0.0, "end": 0.0},
                    success=False,
                )

            log.info("Creating audio chunks...")
            chunks, audio_duration, processing_window = create_chunks(
                speaker_events,
                audio_file_path,
                output_dir,
                window_size_sec,
                skip_start_sec,
                skip_end_sec,
            )

            log.info(f"Creating transcription for {len(chunks)} chunks")
            processed_chunks = []
            for _, chunk in enumerate(chunks):
                chunk_full_path = output_dir / chunk.file_path
                transcription = self.transcriber.transcribe(chunk_full_path.as_posix())
                log.info(
                    f"Transcription for chunk {chunk.start} - {chunk.end}: {transcription}"
                )
                processed_chunks.append(
                    ProcessedAudioChunk(
                        audio_chunk=chunk,
                        transcription=transcription,
                    )
                )

            processed_file = ProcessedAudioFile(
                source_file=audio_file_path.as_posix(),
                output_directory=output_dir.as_posix(),
                audio_duration_sec=audio_duration,
                chunks=processed_chunks,
                success=True,
                processing_window={
                    "start": processing_window[0],
                    "end": processing_window[1],
                },
            )

            # Store data in database
            try:
                audio_file_id = self.audio_store.store_processed_file(
                    processed_file, speaker_embeddings
                )
                log.info(f"Stored audio file data in database with ID: {audio_file_id}")
            except Exception as e:
                log.error(f"Failed to store data in database: {e}")
                # Continue processing even if database storage fails

            log.info(
                f"Completed: {len(chunks)} chunks in {time.time() - start_time:.1f}s"
            )
            return processed_file

        except Exception as e:
            log.error(f"Error processing {audio_file_path.name}: {str(e)}")
            return ProcessedAudioFile(
                source_file=str(audio_file_path),
                output_directory=str(output_dir),
                audio_duration_sec=get_audio_duration(audio_file_path),
                chunks=[],
                processing_window={"start": 0.0, "end": 0.0},
                success=False,
            )

    def upload_chunks_to_gcs(
        self, processed_file: ProcessedAudioFile, job_output_dir: Path
    ) -> bool:
        """
        Upload all chunks to Google Cloud Storage.

        Args:
            processed_file: Processed audio file data
            job_output_dir: Directory containing chunk files

        Returns:
            True if all uploads successful, False otherwise
        """
        try:
            upload_success = True

            for chunk in processed_file.chunks:
                chunk_path = job_output_dir / chunk.audio_chunk.file_path

                if chunk_path.exists():
                    success = self.gcs_client.upload_chunk(
                        chunk_path, chunk.audio_chunk.file_path
                    )
                    if not success:
                        upload_success = False
                        log.error(
                            f"Failed to upload chunk {chunk.audio_chunk.file_path}"
                        )
                else:
                    log.warning(f"Chunk file not found: {chunk_path}")
                    upload_success = False

            return upload_success

        except Exception as e:
            log.error(f"Error uploading chunks to GCS: {e}")
            return False

    def cleanup_temp_dir(self, job_output_dir: Path):
        """
        Clean up temporary directory after processing.

        Args:
            job_output_dir: Directory to clean up
        """
        try:
            if job_output_dir.exists():
                shutil.rmtree(job_output_dir)
                log.info(f"Cleaned up temporary directory: {job_output_dir}")
        except Exception as e:
            log.error(f"Error cleaning up temporary directory {job_output_dir}: {e}")

    def start_listening(self):
        """
        Start listening to the Redis queue for jobs.
        """
        log.info(f"Starting chunker worker, listening to queue: {self.queue_name}")

        # Check Redis health
        if not self.redis_client.health_check():
            log.error("Redis health check failed, exiting")
            return

        while True:
            try:
                # Wait for job from queue
                job_data = self.redis_client.pop_job(self.queue_name, timeout=10)

                if job_data:
                    log.info(f"Received job: {job_data}")
                    success = self.process_job(job_data)

                    if success:
                        log.info("Job processed successfully")
                    else:
                        log.error("Job processing failed")

                # Short sleep to prevent busy waiting
                time.sleep(0.1)

            except KeyboardInterrupt:
                log.info("Received interrupt signal, shutting down worker")
                break
            except Exception as e:
                log.error(f"Error in worker loop: {e}")
                time.sleep(5)  # Wait before retrying


def create_chunks(
    speaker_events: list[SpeakerEvent],
    original_audio_path: Path,
    output_dir: Path,
    window_size_sec: float = 30.0,
    skip_start_sec: float = 120.0,  # 2 minutes
    skip_end_sec: float = 180.0,  # 3 minutes
) -> tuple[list[AudioChunk], float, tuple[float, float]]:
    """
    Create audio chunks from speaker events using a sliding window approach.

    Args:
        speaker_events: List of speaker events in seconds
        original_audio_path: Path to original audio file
        output_dir: Output directory for audio chunks
        window_size_sec: Size of sliding window in seconds
        skip_start_sec: Time to skip from start (will find natural break after this)
        skip_end_sec: Time to skip from end (will find natural break before this)

    Returns:
        Tuple of (chunks, audio_duration, processing_window)
        - chunks: List of AudioChunk objects
        - audio_duration: Total audio duration in seconds
        - processing_window: Tuple of (start_time, end_time) in seconds
    """
    audio_duration = get_audio_duration(original_audio_path)

    # Find natural breaks using speaker events instead of silence detection
    actual_start = find_natural_break_after_time(speaker_events, skip_start_sec)
    actual_end = find_natural_break_after_time(
        speaker_events, audio_duration - skip_end_sec
    )

    # Ensure we don't go beyond the last speaker event
    if speaker_events:
        actual_end = min(actual_end, speaker_events[-1].end)

    log.info(f"Audio duration: {audio_duration:.1f}s")
    log.info(f"Processing from {actual_start:.1f}s to {actual_end:.1f}s")

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

    if not events_in_window:
        log.warning("No speaker events in processing window, skipping chunk creation")
        return [], audio_duration, (actual_start, actual_end)

    chunks = []
    current_chunk_events = []
    current_chunk_duration = 0.0

    for event in events_in_window:
        event_duration = event.end - event.start

        # Check if adding this event would exceed the window size
        # We need to check both the chunk span
        potential_chunk_events = current_chunk_events + [event]
        potential_chunk_start = potential_chunk_events[0].start
        potential_chunk_end = potential_chunk_events[-1].end
        potential_chunk_span = potential_chunk_end - potential_chunk_start

        if potential_chunk_span > window_size_sec and current_chunk_events:
            # Create chunk with current events
            chunk_start = current_chunk_events[0].start
            chunk_end = current_chunk_events[-1].end

            chunk_id = str(ULID())
            chunk_filename = f"{chunk_id}.mp3"
            chunk_path = output_dir / chunk_filename

            extract_audio_segment(
                original_audio_path, chunk_start, chunk_end, chunk_path
            )

            chunk = AudioChunk(
                start=chunk_start * 1000,
                end=chunk_end * 1000,
                file_path=chunk_filename,
                speaker_events=current_chunk_events,
            )

            chunks.append(chunk)
            log.info(
                f"Created chunk {len(chunks)}: {chunk_start:.1f}s-{chunk_end:.1f}s "
                f"({len(current_chunk_events)} events, {current_chunk_duration:.1f}s)"
            )

            # Reset for next chunk
            current_chunk_events = [event]
            current_chunk_duration = event_duration
        else:
            # Add event to current chunk
            current_chunk_events.append(event)
            current_chunk_duration += event_duration

    # Create final chunk if there are remaining events
    if current_chunk_events:
        chunk_start = current_chunk_events[0].start
        chunk_end = current_chunk_events[-1].end
        chunk_span = chunk_end - chunk_start

        if chunk_span > window_size_sec:
            log.warning(
                f"Final chunk {chunk_start:.1f}s-{chunk_end:.1f}s is too long ({chunk_span:.1f}s), skipping"
            )
            return chunks, audio_duration, (actual_start, actual_end)

        chunk_id = str(ULID())
        chunk_filename = f"{chunk_id}.mp3"
        chunk_path = output_dir / chunk_filename

        extract_audio_segment(original_audio_path, chunk_start, chunk_end, chunk_path)

        chunk = AudioChunk(
            start=chunk_start * 1000,
            end=chunk_end * 1000,
            file_path=chunk_filename,
            speaker_events=current_chunk_events,
        )

        chunks.append(chunk)
        log.debug(
            f"Created final chunk {len(chunks)}: {chunk_start:.1f}s-{chunk_end:.1f}s "
            f"({len(current_chunk_events)} events, {current_chunk_duration:.1f}s)"
        )

    return chunks, audio_duration, (actual_start, actual_end)


def queue_audio_file(
    audio_file_path: str,
    queue_name: str = "audio_processing",
    window_size_sec: float = 30.0,
    skip_start_sec: float = 120.0,
    skip_end_sec: float = 180.0,
):
    """
    Queue an audio file for processing.

    Args:
        audio_file_path: Path to the audio file
        queue_name: Redis queue name
        window_size_sec: Size of sliding window in seconds
        skip_start_sec: Time to skip from start
        skip_end_sec: Time to skip from end

    Returns:
        True if successfully queued, False otherwise
    """
    # Check if file has already been processed
    audio_store = AudioStore()
    if audio_store.is_audio_file_processed(audio_file_path):
        log.info(f"Audio file {audio_file_path} has already been processed, skipping")
        return False

    redis_client = get_redis_connection()

    job_data = {
        "audio_file_path": audio_file_path,
        "window_size_sec": window_size_sec,
        "skip_start_sec": skip_start_sec,
        "skip_end_sec": skip_end_sec,
        "timestamp": time.time(),
    }

    redis_client.push_job(queue_name, job_data)
    return True


def queue_audio_batch(
    source_folder: str,
    queue_name: str = "audio_processing",
    window_size_sec: float = 30.0,
    skip_start_sec: float = 120.0,
    skip_end_sec: float = 180.0,
) -> int:
    """
    Queue all audio files in a folder for processing.

    Args:
        source_folder: Path to the source folder containing audio files
        queue_name: Redis queue name
        window_size_sec: Size of sliding window in seconds
        skip_start_sec: Time to skip from start
        skip_end_sec: Time to skip from end

    Returns:
        Number of files queued
    """
    source_path = Path(source_folder)

    if not source_path.exists():
        raise FileNotFoundError(f"Source folder not found: {source_path}")

    audio_files = find_audio_files(source_path)

    if not audio_files:
        log.warning(f"No audio files found in: {source_path}")
        return 0

    log.info(f"Found {len(audio_files)} audio files in: {source_path}")

    # Check which files have already been processed
    audio_store = AudioStore()
    processed_count = 0
    queued_count = 0

    for audio_file in audio_files:
        if audio_store.is_audio_file_processed(str(audio_file)):
            log.info(f"Audio file {audio_file} has already been processed, skipping")
            processed_count += 1
        else:
            success = queue_audio_file(
                str(audio_file),
                queue_name,
                window_size_sec,
                skip_start_sec,
                skip_end_sec,
            )
            if success:
                queued_count += 1

    log.info(
        f"Queued {queued_count} out of {len(audio_files)} audio files ({processed_count} already processed)"
    )
    return queued_count


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Audio chunker with Redis queue support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  worker    - Start a worker to process jobs from Redis queue
  queue     - Queue audio files for processing
  batch     - Queue all audio files in a folder

Examples:
  %(prog)s worker
  %(prog)s queue /path/to/audio/file.mp3
  %(prog)s batch /path/to/audio/folder
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Worker command
    worker_parser = subparsers.add_parser("worker", help="Start worker to process jobs")
    worker_parser.add_argument(
        "--queue-name",
        "-q",
        default="audio_processing",
        help="Redis queue name (default: audio_processing)",
    )
    worker_parser.add_argument(
        "--whisper-checkpoint-path",
        "-c",
        default="pevers/whisperd-nl",
        help="Path to Whisper checkpoint (default: pevers/whisperd-nl)",
    )
    worker_parser.add_argument(
        "--temp-dir",
        "-t",
        default="/tmp/parkiet_chunks",
        help="Temporary directory for processing (default: /tmp/parkiet_chunks)",
    )

    # Queue command
    queue_parser = subparsers.add_parser("queue", help="Queue single audio file")
    queue_parser.add_argument("audio_file", help="Path to audio file")
    queue_parser.add_argument(
        "--queue-name",
        "-q",
        default="audio_processing",
        help="Redis queue name (default: audio_processing)",
    )
    queue_parser.add_argument(
        "--window-size",
        "-s",
        type=float,
        default=30.0,
        help="Size of sliding window in seconds (default: 30.0)",
    )
    queue_parser.add_argument(
        "--skip-start",
        type=float,
        default=60.0,
        help="Time to skip from start in seconds (default: 60.0)",
    )
    queue_parser.add_argument(
        "--skip-end",
        type=float,
        default=120.0,
        help="Time to skip from end in seconds (default: 120.0)",
    )

    # Batch command
    batch_parser = subparsers.add_parser(
        "batch", help="Queue all audio files in folder"
    )
    batch_parser.add_argument("source_folder", help="Path to source folder")
    batch_parser.add_argument(
        "--queue-name",
        "-q",
        default="audio_processing",
        help="Redis queue name (default: audio_processing)",
    )
    batch_parser.add_argument(
        "--window-size",
        "-s",
        type=float,
        default=30.0,
        help="Size of sliding window in seconds (default: 30.0)",
    )
    batch_parser.add_argument(
        "--skip-start",
        type=float,
        default=120.0,
        help="Time to skip from start in seconds (default: 120.0)",
    )
    batch_parser.add_argument(
        "--skip-end",
        type=float,
        default=180.0,
        help="Time to skip from end in seconds (default: 180.0)",
    )

    args = parser.parse_args()

    if args.command == "worker":
        worker = ChunkerWorker(
            queue_name=args.queue_name,
            whisper_checkpoint_path=args.whisper_checkpoint_path,
            temp_dir=args.temp_dir,
        )
        worker.start_listening()

    elif args.command == "queue":
        success = queue_audio_file(
            args.audio_file,
            args.queue_name,
            args.window_size,
            args.skip_start,
            args.skip_end,
        )
        if success:
            log.info(f"Successfully queued: {args.audio_file}")
        else:
            log.info(f"File {args.audio_file} was already processed, not queued")
    elif args.command == "batch":
        count = queue_audio_batch(
            args.source_folder,
            args.queue_name,
            args.window_size,
            args.skip_start,
            args.skip_end,
        )
        log.info(f"Queued {count} files")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
