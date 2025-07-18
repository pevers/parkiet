import argparse
import logging
import random
from pathlib import Path
import statistics
import time
import torch
import shutil
import warnings
from dotenv import load_dotenv
import concurrent.futures

warnings.filterwarnings(
    "ignore", message="The MPEG_LAYER_III subtype is unknown to TorchAudio"
)

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
    extract_audio_segments_parallel,
    find_natural_break_after_time,
)
from parkiet.audioprep.speaker_extractor import SpeakerExtractor
from parkiet.audioprep.transcriber import Transcriber, WhisperTimestampedTranscriber
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
        gpu_id: int = 0,
        max_workers: int = 4,
    ):
        """
        Initialize the chunker worker.

        Args:
            queue_name: Redis queue name to listen to
            whisper_checkpoint_path: Path to Whisper checkpoint
            temp_dir: Temporary directory for processing chunks
            gpu_id: GPU ID to use for processing (default: 0)
            max_workers: Maximum number of parallel workers for audio extraction (default: 4)
        """
        self.queue_name = queue_name
        self.max_workers = max_workers
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        # Initialize services
        self.redis_client = get_redis_connection()
        self.gcs_client = get_gcs_client()
        self.audio_store = AudioStore()

        # Initialize AI models
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{gpu_id}")
            torch.cuda.set_device(gpu_id)
        else:
            device = torch.device("cpu")

        self.speaker_extractor = SpeakerExtractor(device.type)
        self.transcriber = Transcriber(whisper_checkpoint_path, device)
        self.timestamped_transcriber = WhisperTimestampedTranscriber(
            "openai/whisper-large-v3-turbo", device.type
        )

    def download_from_gcs(self, gcs_path: str, local_path: Path) -> bool:
        """
        Download a file from GCS to local path.

        Args:
            gcs_path: GCS path to download from
            local_path: Local path to download to

        Returns:
            True if successful, False otherwise
        """
        try:
            blob = self.gcs_client.bucket.blob(gcs_path)
            blob.download_to_filename(str(local_path))
            log.info(f"Downloaded {gcs_path} to {local_path}")
            return True
        except Exception as e:
            log.error(f"Failed to download {gcs_path} from GCS: {e}")
            return False

    def process_job(self, job_data: dict) -> bool:
        """
        Process a single job from the queue.

        Args:
            job_data: Job data containing GCS audio file path and processing parameters

        Returns:
            True if successful, False otherwise
        """
        try:
            # Extract job parameters
            gcs_audio_path = job_data["audio_file_path"]
            window_size_sec = job_data.get("window_size_sec", 30.0)
            skip_start_sec = job_data.get("skip_start_sec", 120.0)
            skip_end_sec = job_data.get("skip_end_sec", 180.0)

            log.info(f"Processing job: {gcs_audio_path}")

            # Check if file has already been processed
            if self.audio_store.is_audio_file_processed(gcs_audio_path):
                log.info(
                    f"Audio file {gcs_audio_path} has already been processed, skipping"
                )
                return True

            # Create temporary output directory for this job
            job_id = str(ULID())
            job_output_dir = self.temp_dir / job_id
            job_output_dir.mkdir(parents=True, exist_ok=True)

            try:
                # Download from GCS to temporary location
                gcs_filename = Path(gcs_audio_path).name
                local_audio_path = job_output_dir / gcs_filename

                if not self.download_from_gcs(gcs_audio_path, local_audio_path):
                    log.error(f"Failed to download {gcs_audio_path} from GCS")
                    return False

                # Process the audio file
                processed_file = self.process_single_audio_file(
                    local_audio_path,
                    job_output_dir,
                    gcs_audio_path,
                    window_size_sec,
                    skip_start_sec,
                    skip_end_sec,
                )

                # Update the source file path to the original GCS path
                processed_file.source_file = gcs_audio_path

                if processed_file.success:
                    upload_success = self.upload_chunks_to_gcs(
                        processed_file, job_output_dir
                    )

                    if upload_success:
                        log.info(
                            f"Successfully processed and uploaded chunks for {gcs_audio_path}"
                        )
                        return True
                    else:
                        log.error(f"Failed to upload chunks for {gcs_audio_path}")
                        return False
                else:
                    log.error(f"Failed to process audio file {gcs_audio_path}")
                    return False

            finally:
                # Clean up temporary directory
                self.cleanup_temp_dir(job_output_dir)

                # Clear CUDA cache after each job to prevent memory accumulation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        except Exception as e:
            log.error(f"Error processing job: {e}")
            return False

    def process_single_audio_file(
        self,
        audio_file_path: Path,
        output_dir: Path,
        gcs_audio_path: str,
        window_size_sec: float = 30.0,
        skip_start_sec: float = 120.0,
        skip_end_sec: float = 180.0,
    ) -> ProcessedAudioFile:
        """
        Process a single audio file.

        Args:
            audio_file_path: Path to the audio file
            output_dir: Output directory for chunks
            gcs_audio_path: GCS path to the audio file
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
                    gcs_audio_path=gcs_audio_path,
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
                gcs_audio_path,
                window_size_sec,
                skip_start_sec,
                skip_end_sec,
                max_workers=self.max_workers,
            )

            log.info(f"Creating transcription for {len(chunks)} chunks")
            processed_chunks = []
            for _, chunk in enumerate(chunks):
                chunk_full_path = output_dir / chunk.file_path
                transcription = self.transcriber.transcribe_with_confidence(chunk_full_path.as_posix())

                # Get timestamped transcription with speaker tags
                timestamped_result = (
                    self.timestamped_transcriber.transcribe_with_timestamps(
                        chunk_full_path.as_posix()
                    )
                )
                # Convert chunk.start from milliseconds to seconds for timing alignment
                chunk_start_sec = chunk.start / 1000.0
                transcription_clean = self._add_speaker_tags(
                    timestamped_result, chunk.speaker_events, chunk_start_sec
                )

                log.info(
                    f"\nTranscription for chunk {chunk.start} - {chunk.end}:\n{transcription}\n"
                )
                log.info(f"\nClean transcription:\n{transcription_clean} {timestamped_result['confidence']}\n")
                processed_chunks.append(
                    ProcessedAudioChunk(
                        audio_chunk=chunk,
                        transcription=transcription["text"],
                        transcription_conf=transcription["confidence"],
                        transcription_clean=transcription_clean,
                        transcription_clean_conf=timestamped_result["confidence"],
                    )
                )

            processed_file = ProcessedAudioFile(
                source_file=audio_file_path.as_posix(),
                output_directory=output_dir.as_posix(),
                gcs_audio_path=gcs_audio_path,
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
                gcs_audio_path=gcs_audio_path,
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

                # Use the GCS path already set in the chunk
                gcs_chunk_path = chunk.audio_chunk.gcs_file_path

                if chunk_path.exists():
                    success = self.gcs_client.upload_chunk(chunk_path, gcs_chunk_path)
                    if not success:
                        upload_success = False
                        log.error(f"Failed to upload chunk {gcs_chunk_path}")
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

    def _add_speaker_tags(
        self,
        timestamped_result: dict,
        speaker_events: list[SpeakerEvent],
        chunk_start_sec: float,
    ) -> str:
        """Add speaker tags to timestamped transcription segments based on actual speaker events."""
        if not timestamped_result.get("segments"):
            return timestamped_result.get("text", "")

        # Create a mapping of speakers in order of first appearance in this chunk
        speaker_mapping = {}
        speaker_counter = 1

        transcript_parts = []
        current_speaker = None

        for segment in timestamped_result["segments"]:
            segment_text = segment["text"].strip()
            if not segment_text:
                continue

            # Get segment timing (whisper timestamps are relative to chunk start)
            segment_start = segment.get("start", 0)
            segment_end = segment.get("end", segment_start)

            # Convert to absolute time by adding chunk start time
            absolute_segment_start = chunk_start_sec + segment_start
            absolute_segment_end = chunk_start_sec + segment_end
            absolute_segment_mid = (absolute_segment_start + absolute_segment_end) / 2

            # Find the speaker for this segment based on overlap with speaker events
            segment_speaker = None
            for event in speaker_events:
                # Check if segment overlaps with this speaker event
                if (
                    absolute_segment_start < event.end
                    and absolute_segment_end > event.start
                ) or (event.start <= absolute_segment_mid <= event.end):
                    segment_speaker = event.speaker
                    break

            # Only add speaker tag if speaker changes or it's the first segment
            if segment_speaker and segment_speaker != current_speaker:
                # Create mapping for new speakers as they appear
                if segment_speaker not in speaker_mapping:
                    speaker_mapping[segment_speaker] = f"S{speaker_counter}"
                    speaker_counter += 1

                # Map to relative speaker tag (S1, S2, etc.)
                relative_speaker = speaker_mapping[segment_speaker]
                speaker_tag = f"[{relative_speaker}]"
                transcript_parts.append(f"{speaker_tag} {segment_text}")
                current_speaker = segment_speaker
            else:
                # Same speaker, just add the text
                transcript_parts.append(segment_text)

        return " ".join(transcript_parts)


def create_chunks(
    speaker_events: list[SpeakerEvent],
    original_audio_path: Path,
    output_dir: Path,
    gcs_audio_path: str,
    window_size_sec: float = 30.0,
    skip_start_sec: float = 100.0,
    skip_end_sec: float = 100.0,
    max_workers: int = 4,
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
        gcs_audio_path: GCS path to the original audio file (used for chunk GCS paths)
        max_workers: Maximum number of parallel workers for audio extraction (default: 4)

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
    chunk_tasks = []  # List to store audio extraction tasks for parallel processing
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

            # Add task to the list for parallel extraction
            chunk_tasks.append(
                (original_audio_path, chunk_start, chunk_end, chunk_path)
            )

            # Create GCS path for the chunk
            original_file_path = Path(gcs_audio_path)
            original_file_name = original_file_path.stem
            gcs_chunk_path = f"{original_file_name}/{chunk_filename}"

            chunk = AudioChunk(
                start=chunk_start * 1000,
                end=chunk_end * 1000,
                file_path=chunk_filename,
                speaker_events=current_chunk_events,
                gcs_file_path=gcs_chunk_path,
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

        # Add task to the list for parallel extraction
        chunk_tasks.append((original_audio_path, chunk_start, chunk_end, chunk_path))

        # Create GCS path for the chunk
        original_file_path = Path(gcs_audio_path)
        original_file_name = original_file_path.stem
        gcs_chunk_path = f"{original_file_name}/{chunk_filename}"

        chunk = AudioChunk(
            start=chunk_start * 1000,
            end=chunk_end * 1000,
            file_path=chunk_filename,
            speaker_events=current_chunk_events,
            gcs_file_path=gcs_chunk_path,
        )

        chunks.append(chunk)
        log.debug(
            f"Created final chunk {len(chunks)}: {chunk_start:.1f}s-{chunk_end:.1f}s "
            f"({len(current_chunk_events)} events, {current_chunk_duration:.1f}s)"
        )

    # Extract all audio segments in parallel
    if chunk_tasks:
        log.info(
            f"Extracting {len(chunk_tasks)} audio chunks in parallel with {max_workers} workers"
        )
        extract_audio_segments_parallel(chunk_tasks, max_workers)
        log.info("Completed parallel audio extraction")

    return chunks, audio_duration, (actual_start, actual_end)


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
    mp3_file_list: str,
    queue_name: str = "audio_processing",
    window_size_sec: float = 30.0,
    skip_start_sec: float = 120.0,
    skip_end_sec: float = 180.0,
) -> int:
    """
    Queue GCS audio files from a file list for processing.

    Args:
        mp3_file_list: Path to file containing GCS MP3 paths (one per line)
        queue_name: Redis queue name
        window_size_sec: Size of sliding window in seconds
        skip_start_sec: Time to skip from start
        skip_end_sec: Time to skip from end

    Returns:
        Number of files queued
    """
    file_path = Path(mp3_file_list)

    if not file_path.exists():
        raise FileNotFoundError(f"MP3 file list not found: {file_path}")

    # Read GCS MP3 file paths from file
    with open(file_path, "r", encoding="utf-8") as f:
        gcs_mp3_paths = [line.strip() for line in f if line.strip()]

    if not gcs_mp3_paths:
        log.warning(f"No GCS MP3 paths found in: {file_path}")
        return 0

    log.info(f"Found {len(gcs_mp3_paths)} GCS MP3 paths in: {file_path}")

    # Random shuffle the list because we will probably not be able to process all files
    random.shuffle(gcs_mp3_paths)

    # Check which files have already been processed
    audio_store = AudioStore()
    processed_count = 0
    queued_count = 0

    for gcs_mp3_path in gcs_mp3_paths:
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


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Audio chunker with Redis queue support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  worker    - Start a worker to process jobs from Redis queue
  queue     - Queue a single GCS audio file for processing
  file      - Queue GCS audio files from a file list

Examples:
  %(prog)s worker
  %(prog)s queue gs://bucket/path/to/audio/file.mp3
  %(prog)s file bucket_mp3_files.txt
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
    worker_parser.add_argument(
        "--gpu-id",
        "-g",
        type=int,
        default=0,
        help="GPU ID to use for processing (default: 0)",
    )
    worker_parser.add_argument(
        "--max-workers",
        "-w",
        type=int,
        default=4,
        help="Maximum number of parallel workers for audio extraction (default: 4)",
    )

    # Queue command
    queue_parser = subparsers.add_parser("queue", help="Queue single GCS audio file")
    queue_parser.add_argument("gcs_audio_path", help="GCS path to audio file")
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

    # File command
    file_parser = subparsers.add_parser(
        "file", help="Queue GCS audio files from file list"
    )
    file_parser.add_argument(
        "mp3_file_list", help="Path to file containing GCS MP3 paths"
    )
    file_parser.add_argument(
        "--queue-name",
        "-q",
        default="audio_processing",
        help="Redis queue name (default: audio_processing)",
    )
    file_parser.add_argument(
        "--window-size",
        "-s",
        type=float,
        default=30.0,
        help="Size of sliding window in seconds (default: 30.0)",
    )
    file_parser.add_argument(
        "--skip-start",
        type=float,
        default=120.0,
        help="Time to skip from start in seconds (default: 120.0)",
    )
    file_parser.add_argument(
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
            gpu_id=args.gpu_id,
            max_workers=args.max_workers,
        )
        worker.start_listening()

    elif args.command == "queue":
        success = queue_audio_file(
            args.gcs_audio_path,
            args.queue_name,
            args.window_size,
            args.skip_start,
            args.skip_end,
        )
        if success:
            log.info(f"Successfully queued: {args.gcs_audio_path}")
        else:
            log.info(f"File {args.gcs_audio_path} was already processed, not queued")
    elif args.command == "file":
        count = queue_from_file(
            args.mp3_file_list,
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
