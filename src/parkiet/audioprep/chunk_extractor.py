import argparse
import logging
import time
import torch
import shutil
import warnings
from pathlib import Path
from dotenv import load_dotenv
from ulid import ULID

warnings.filterwarnings(
    "ignore", message="The MPEG_LAYER_III subtype is unknown to TorchAudio"
)

load_dotenv()
from parkiet.audioprep.schemas import (
    AudioChunk,
    SpeakerEvent,
    ProcessedAudioFile,
)
from parkiet.utils.audio import (
    get_audio_duration,
    extract_audio_segments_parallel,
    find_natural_break_after_time,
    validate_audio_file,
    convert_to_wav,
)
from parkiet.audioprep.speaker_extractor import SpeakerExtractor
from parkiet.database.audio_store import AudioStore
from parkiet.database.redis_connection import get_redis_connection
from parkiet.storage.gcs_client import get_gcs_client

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)


class ChunkExtractor:
    """Extract audio chunks and queue them for transcription."""

    def __init__(
        self,
        input_queue_name: str = "audio_processing",
        output_queue_name: str = "transcription_queue",
        chunks_dir: str = "/tmp/parkiet_chunks",
        gpu_id: int = 0,
        max_workers: int = 4,
    ):
        self.input_queue_name = input_queue_name
        self.output_queue_name = output_queue_name
        self.max_workers = max_workers
        self.chunks_dir = Path(chunks_dir)
        self.chunks_dir.mkdir(parents=True, exist_ok=True)

        # Initialize services
        self.redis_client = get_redis_connection()
        self.gcs_client = get_gcs_client()
        self.audio_store = AudioStore()

        # Initialize device configuration
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{gpu_id}")
            torch.cuda.set_device(gpu_id)
        else:
            self.device = torch.device("cpu")

        # Load speaker extractor model at initialization and keep it loaded
        log.info("Loading speaker extractor model...")
        self.speaker_extractor = SpeakerExtractor(str(self.device))
        log.info("Speaker extractor model loaded successfully")

    def download_from_gcs(self, gcs_path: str, local_path: Path) -> bool:
        """Download a file from GCS to local path."""
        try:
            blob = self.gcs_client.bucket.blob(gcs_path)
            blob.download_to_filename(str(local_path))
            log.info(f"Downloaded {gcs_path} to {local_path}")
            return True
        except Exception as e:
            log.exception(f"Failed to download {gcs_path} from GCS")
            return False

    def extract_chunks(self, job_data: dict) -> bool:
        """Extract chunks from audio file and queue them for transcription."""
        try:
            # Extract job parameters
            gcs_audio_path = job_data["audio_file_path"]
            window_size_sec = job_data.get("window_size_sec", 30.0)
            skip_start_sec = job_data.get("skip_start_sec", 80.0)
            skip_end_sec = job_data.get("skip_end_sec", 80.0)

            log.info(f"Extracting chunks from: {gcs_audio_path}")

            # Check if file has already been processed
            if self.audio_store.is_audio_file_processed(gcs_audio_path):
                log.info(f"Audio file {gcs_audio_path} already processed, skipping")
                return True

            # Create temporary directory for this job
            job_id = str(ULID())
            job_temp_dir = self.chunks_dir / "temp" / job_id
            job_temp_dir.mkdir(parents=True, exist_ok=True)

            # Create permanent chunks directory
            job_chunks_dir = self.chunks_dir / "chunks" / job_id
            job_chunks_dir.mkdir(parents=True, exist_ok=True)

            try:
                # Download from GCS to temporary location
                gcs_filename = Path(gcs_audio_path).name
                local_audio_path = job_temp_dir / gcs_filename

                if not self.download_from_gcs(gcs_audio_path, local_audio_path):
                    log.error(f"Failed to download {gcs_audio_path}")
                    return False

                # Validate and convert audio file
                if not validate_audio_file(local_audio_path):
                    log.error(f"Audio file {gcs_audio_path} is invalid")
                    return False

                wav_filename = f"{Path(gcs_filename).stem}.wav"
                wav_audio_path = job_temp_dir / wav_filename

                try:
                    convert_to_wav(local_audio_path, wav_audio_path)
                    local_audio_path.unlink()  # Clean up original
                    log.info(f"Converted {gcs_filename} to WAV format")
                except Exception as e:
                    log.error(f"Failed to convert {gcs_filename} to WAV: {e}")
                    return False

                # Extract speaker events using pre-loaded model
                speaker_events, speaker_embeddings = (
                    self.speaker_extractor.extract_speaker_events(wav_audio_path)
                )
                log.info(f"Found {len(speaker_events)} speaker events")

                if not speaker_events:
                    log.warning("No speaker events found, skipping")
                    return False

                # Create chunks
                chunks, audio_duration, processing_window = self._create_chunks(
                    speaker_events,
                    wav_audio_path,
                    job_chunks_dir,
                    gcs_audio_path,
                    window_size_sec,
                    skip_start_sec,
                    skip_end_sec,
                )

                # Convert numpy arrays to lists for JSON serialization
                serializable_embeddings = {}
                for speaker, embedding in speaker_embeddings.items():
                    serializable_embeddings[speaker] = embedding.tolist()

                # Store audio file in database first
                processed_file = ProcessedAudioFile(
                    source_file=gcs_audio_path,
                    gcs_audio_path=gcs_audio_path,
                    audio_duration_sec=audio_duration,
                    chunks=[],  # Empty for now, will be filled by transcriber
                    success=True,
                    processing_window={
                        "start": processing_window[0],
                        "end": processing_window[1],
                    },
                )

                audio_file_id = self.audio_store.store_processed_file(
                    processed_file, serializable_embeddings
                )
                log.info(f"Stored audio file in database with ID: {audio_file_id}")

                # Queue chunks for transcription
                for chunk in chunks:
                    chunk_message = {
                        "job_id": job_id,
                        "chunk_id": chunk.file_path,
                        "chunk_path": str(job_chunks_dir / chunk.file_path),
                        "gcs_audio_path": gcs_audio_path,
                        "audio_file_id": audio_file_id,
                        "chunk_data": chunk.model_dump(),
                        "speaker_embeddings": serializable_embeddings,
                        "timestamp": time.time(),
                    }
                    self.redis_client.push_job(self.output_queue_name, chunk_message)

                log.info(f"Queued {len(chunks)} chunks for transcription")
                return True

            finally:
                # Clean up temporary directory
                if job_temp_dir.exists():
                    shutil.rmtree(job_temp_dir)
                    log.info(f"Cleaned up temp directory: {job_temp_dir}")

        except Exception as e:
            log.exception("Error extracting chunks")
            return False

    def _create_chunks(
        self,
        speaker_events: list[SpeakerEvent],
        audio_file_path: Path,
        output_dir: Path,
        gcs_audio_path: str,
        window_size_sec: float,
        skip_start_sec: float,
        skip_end_sec: float,
    ) -> tuple[list[AudioChunk], float, tuple[float, float]]:
        """Create audio chunks from speaker events."""
        audio_duration = get_audio_duration(audio_file_path)

        # Find natural breaks
        actual_start = find_natural_break_after_time(speaker_events, skip_start_sec)
        actual_end = find_natural_break_after_time(
            speaker_events, audio_duration - skip_end_sec
        )

        if speaker_events:
            actual_end = min(actual_end, speaker_events[-1].end)

        log.info(f"Processing from {actual_start:.1f}s to {actual_end:.1f}s")

        # Filter events to processing window
        events_in_window = []
        for event in speaker_events:
            if event.end > actual_start and event.start < actual_end:
                clipped_start = max(event.start, actual_start)
                clipped_end = min(event.end, actual_end)
                events_in_window.append(
                    SpeakerEvent(
                        start=clipped_start, end=clipped_end, speaker=event.speaker
                    )
                )

        if not events_in_window:
            log.warning("No events in processing window")
            return [], audio_duration, (actual_start, actual_end)

        chunks = []
        chunk_tasks = []
        current_chunk_events = []

        for event in events_in_window:
            event_duration = event.end - event.start

            # Skip events longer than window size
            if event_duration > window_size_sec:
                log.warning(f"Skipping long event: {event_duration:.1f}s")
                continue

            # Check if adding event would exceed window size
            potential_chunk_events = current_chunk_events + [event]
            if len(potential_chunk_events) > 0:
                potential_chunk_span = (
                    potential_chunk_events[-1].end - potential_chunk_events[0].start
                )

                if potential_chunk_span > window_size_sec and current_chunk_events:
                    # Create chunk with current events
                    chunk_start = current_chunk_events[0].start
                    chunk_end = current_chunk_events[-1].end
                    chunk_span = chunk_end - chunk_start

                    if chunk_span >= 1.0:  # Min 1 second chunks
                        chunk_id = str(ULID())
                        chunk_filename = f"{chunk_id}.wav"
                        chunk_path = output_dir / chunk_filename

                        # Add to extraction tasks
                        chunk_tasks.append(
                            (audio_file_path, chunk_start, chunk_end, chunk_path)
                        )

                        # Create GCS path
                        original_file_path = Path(gcs_audio_path)
                        parent_dir = original_file_path.parent.name
                        original_file_name = original_file_path.stem
                        gcs_chunk_path = (
                            f"{parent_dir}/{original_file_name}/{chunk_filename}"
                        )

                        chunk = AudioChunk(
                            start=chunk_start * 1000,
                            end=chunk_end * 1000,
                            file_path=chunk_filename,
                            speaker_events=current_chunk_events,
                            gcs_file_path=gcs_chunk_path,
                        )

                        chunks.append(chunk)
                        log.info(
                            f"Created chunk {len(chunks)}: {chunk_start:.1f}s-{chunk_end:.1f}s"
                        )

                    # Reset for next chunk
                    current_chunk_events = [event]
                else:
                    # Add event to current chunk
                    current_chunk_events.append(event)

        # Create final chunk
        if current_chunk_events:
            chunk_start = current_chunk_events[0].start
            chunk_end = current_chunk_events[-1].end
            chunk_span = chunk_end - chunk_start

            if chunk_span >= 1.0 and chunk_span <= window_size_sec:
                chunk_id = str(ULID())
                chunk_filename = f"{chunk_id}.wav"
                chunk_path = output_dir / chunk_filename

                chunk_tasks.append(
                    (audio_file_path, chunk_start, chunk_end, chunk_path)
                )

                original_file_path = Path(gcs_audio_path)
                parent_dir = original_file_path.parent.name
                original_file_name = original_file_path.stem
                gcs_chunk_path = f"{parent_dir}/{original_file_name}/{chunk_filename}"

                chunk = AudioChunk(
                    start=chunk_start * 1000,
                    end=chunk_end * 1000,
                    file_path=chunk_filename,
                    speaker_events=current_chunk_events,
                    gcs_file_path=gcs_chunk_path,
                )

                chunks.append(chunk)
                log.info(f"Created final chunk: {chunk_start:.1f}s-{chunk_end:.1f}s")

        # Extract all audio segments in parallel
        if chunk_tasks:
            log.info(
                f"Extracting {len(chunk_tasks)} chunks with {self.max_workers} workers"
            )
            extract_audio_segments_parallel(chunk_tasks, self.max_workers)
            log.info("Completed parallel audio extraction")

        return chunks, audio_duration, (actual_start, actual_end)

    def start_listening(self):
        """Start listening to the input queue for extraction jobs."""
        log.info(f"Starting chunk extractor, listening to: {self.input_queue_name}")

        if not self.redis_client.health_check():
            log.error("Redis health check failed")
            return

        while True:
            try:
                job_data = self.redis_client.pop_job(self.input_queue_name, timeout=10)

                if job_data:
                    log.info(
                        f"Received job for: {job_data.get('audio_file_path', 'unknown')}"
                    )
                    success = self.extract_chunks(job_data)

                    if success:
                        log.info("Job processed successfully")
                    else:
                        log.error("Job processing failed")

                time.sleep(0.1)

            except KeyboardInterrupt:
                log.info("Shutting down chunk extractor")
                break
            except Exception as e:
                log.exception("Error in extractor loop")
                time.sleep(5)


def main():
    """Main function for chunk extractor."""
    parser = argparse.ArgumentParser(description="Audio chunk extractor")

    parser.add_argument(
        "--input-queue",
        default="audio_processing",
        help="Input queue name for audio files",
    )
    parser.add_argument(
        "--output-queue",
        default="transcription_queue",
        help="Output queue name for chunks",
    )
    parser.add_argument(
        "--chunks-dir",
        default="/data/chunk_writer",
        help="Directory for storing chunks",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="GPU ID for processing",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Max parallel workers for audio extraction",
    )

    args = parser.parse_args()

    extractor = ChunkExtractor(
        input_queue_name=args.input_queue,
        output_queue_name=args.output_queue,
        chunks_dir=args.chunks_dir,
        gpu_id=args.gpu_id,
        max_workers=args.max_workers,
    )
    extractor.start_listening()


if __name__ == "__main__":
    main()
