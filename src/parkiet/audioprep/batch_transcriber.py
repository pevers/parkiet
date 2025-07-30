import argparse
import logging
import time
import torch
import re
import shutil
import multiprocessing
import signal
import sys
from pathlib import Path
from dotenv import load_dotenv
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import librosa

load_dotenv()
from parkiet.audioprep.schemas import AudioChunk, ProcessedAudioChunk
from parkiet.database.audio_store import AudioStore
from parkiet.database.redis_connection import get_redis_connection
from parkiet.storage.gcs_client import get_gcs_client

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)


class BatchTranscriber:
    """Batch transcriber that processes chunks from queue."""

    def __init__(
        self,
        input_queue_name: str = "transcription_queue",
        whisper_checkpoint_path: str = "pevers/whisperd-nl",
        whisper_clean_checkpoint_path: str = "pevers/whisper-clean-nl",
        gpu_id: int = 0,
        batch_size: int = 8,
        inference_batch_size: int = 4,
        max_chunk_duration: float = 30.0,
        chunks_dir: str = "data/chunks",
        upload_to_gcs: bool = False,
        dtype: torch.dtype = torch.float16,
    ):
        self.input_queue_name = input_queue_name
        self.batch_size = batch_size
        self.inference_batch_size = inference_batch_size
        self.max_chunk_duration = max_chunk_duration
        self.chunks_dir = Path(chunks_dir)
        self.upload_to_gcs = upload_to_gcs

        # Only create local directory if not uploading to GCS
        if not self.upload_to_gcs:
            self.chunks_dir.mkdir(parents=True, exist_ok=True)

        # Initialize services
        self.redis_client = get_redis_connection()
        self.audio_store = AudioStore()

        # Initialize GCS client if needed
        if self.upload_to_gcs:
            self.gcs_client = get_gcs_client()

        # Initialize device configuration
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{gpu_id}")
            torch.cuda.set_device(gpu_id)
        else:
            self.device = torch.device("cpu")

        # Load models at initialization and keep them loaded
        log.info(f"Loading transcription models with dtype: {dtype}...")
        self.transcriber = WhisperTranscriber(
            whisper_checkpoint_path, self.device, self.inference_batch_size, dtype
        )
        self.clean_transcriber = WhisperTranscriber(
            whisper_clean_checkpoint_path, self.device, self.inference_batch_size, dtype
        )
        log.info("Transcription models loaded successfully")

        self.pending_chunks = []

    def process_chunk_batch(self, chunk_messages: list[dict]) -> bool:
        """Process a batch of chunks."""
        try:
            log.info(f"Processing batch of {len(chunk_messages)} chunks")

            # Filter chunks by duration
            valid_chunks = []
            for chunk_msg in chunk_messages:
                chunk_data = chunk_msg["chunk_data"]
                duration_ms = chunk_data["end"] - chunk_data["start"]
                duration_sec = duration_ms / 1000.0

                if duration_sec <= self.max_chunk_duration:
                    valid_chunks.append(chunk_msg)
                else:
                    log.warning(
                        f"Skipping chunk {chunk_data['file_path']} - too long: {duration_sec:.1f}s"
                    )

            if not valid_chunks:
                log.warning("No valid chunks in batch")
                return True

            # Load audio data once for both models
            audio_data = []
            target_sr = 16000
            for chunk_msg in valid_chunks:
                chunk_path = chunk_msg["chunk_path"]
                if Path(chunk_path).exists():
                    try:
                        audio, sr = librosa.load(chunk_path, sr=target_sr)
                        if sr != target_sr:
                            audio = librosa.resample(
                                audio, orig_sr=sr, target_sr=target_sr
                            )
                        audio_data.append((audio, chunk_path))
                    except Exception as e:
                        log.error(f"Failed to load audio {chunk_path}: {e}")
                else:
                    log.warning(f"Chunk file not found: {chunk_path}")

            if not audio_data:
                log.warning("No valid audio files found")
                return False

            # Batch transcribe with both models using pre-loaded audio
            log.info(
                f"Running batch transcription with whisperd-nl model (inference batch size: {self.inference_batch_size})"
            )
            whisperd_results = self.transcriber.batch_transcribe(audio_data)

            log.info(
                f"Running batch transcription with whisper-clean-nl model (inference batch size: {self.inference_batch_size})"
            )
            clean_results = self.clean_transcriber.batch_transcribe(audio_data)

            # Process results and store in database
            processed_chunks = []
            for i, chunk_msg in enumerate(valid_chunks):
                if i < len(whisperd_results) and i < len(clean_results):
                    chunk_data = AudioChunk(**chunk_msg["chunk_data"])

                    processed_chunk = ProcessedAudioChunk(
                        audio_chunk=chunk_data,
                        transcription=whisperd_results[i]["text"],
                        transcription_clean=clean_results[i]["text"]
                    )

                    processed_chunks.append(processed_chunk)

                    log.info(f"Processed chunk {chunk_data.file_path}")
                    log.info(f"Whisperd: {whisperd_results[i]['text']}...")
                    log.info(f"Clean: {clean_results[i]['text']}...")

            # Store processed chunks in database
            if processed_chunks:
                try:
                    # Group by audio_file_id for database storage
                    jobs_data = {}
                    for i, chunk_msg in enumerate(valid_chunks):
                        if i < len(processed_chunks):
                            audio_file_id = chunk_msg["audio_file_id"]
                            if audio_file_id not in jobs_data:
                                jobs_data[audio_file_id] = {
                                    "speaker_embeddings": chunk_msg[
                                        "speaker_embeddings"
                                    ],
                                    "chunks": [],
                                }
                            jobs_data[audio_file_id]["chunks"].append(
                                processed_chunks[i]
                            )

                    # Add chunks to each audio file
                    for audio_file_id, job_data in jobs_data.items():
                        chunk_ids = self.audio_store.add_chunks_to_audio_file(
                            audio_file_id,
                            job_data["chunks"],
                            job_data["speaker_embeddings"],
                        )
                        log.info(
                            f"Added {len(chunk_ids)} chunks to audio file ID: {audio_file_id}"
                        )

                except Exception as e:
                    log.error(f"Failed to store processed chunks in database: {e}")

            log.info(f"Successfully processed batch of {len(processed_chunks)} chunks")

            # Move processed chunks to designated folder for later GCS upload
            self._move_chunks_to_upload_folder(valid_chunks)

            return True

        except Exception as e:
            log.error(f"Error processing chunk batch: {e}")
            return False

    def _move_chunks_to_upload_folder(self, chunk_messages: list[dict]) -> None:
        """Move processed chunks to the designated folder for later GCS upload or upload directly to GCS."""
        for chunk_msg in chunk_messages:
            chunk_path = Path(chunk_msg["chunk_path"])
            if not chunk_path.exists():
                log.warning(f"Chunk file not found for processing: {chunk_path}")
                continue

            gcs_audio_path = chunk_msg["gcs_audio_path"]
            original_file_path = Path(gcs_audio_path)
            parent_dir = original_file_path.parent.name
            original_file_name = original_file_path.stem
            chunk_filename = chunk_path.name

            if self.upload_to_gcs:
                # Upload directly to GCS
                self._upload_chunk_to_gcs(
                    chunk_path, parent_dir, original_file_name, chunk_filename
                )
            else:
                # Move to local directory structure
                self._move_chunk_locally(
                    chunk_path, parent_dir, original_file_name, chunk_filename
                )

    def _upload_chunk_to_gcs(
        self,
        chunk_path: Path,
        parent_dir: str,
        original_file_name: str,
        chunk_filename: str,
    ) -> None:
        """Upload a chunk directly to GCS."""
        # Create GCS path: chunks/{parent_dir}/{original_file_name}/{chunk_filename}
        gcs_path = f"chunks/{parent_dir}/{original_file_name}/{chunk_filename}"

        # Check if file already exists in GCS
        if self.gcs_client.file_exists(gcs_path):
            log.warning(f"Chunk {chunk_filename} already exists in GCS at {gcs_path}")
            # Remove the local file since it's already in GCS
            if chunk_path.exists():
                chunk_path.unlink()
                log.info(
                    f"Removed local chunk {chunk_filename} (already exists in GCS)"
                )
            return

        # Upload to GCS
        if self.gcs_client.upload_file(chunk_path, gcs_path):
            log.info(f"Uploaded chunk {chunk_filename} to GCS: {gcs_path}")
            # Remove local file after successful upload
            if chunk_path.exists():
                chunk_path.unlink()
                log.info(
                    f"Removed local chunk {chunk_filename} after successful upload"
                )
        else:
            log.error(f"Failed to upload chunk {chunk_filename} to GCS")
            # Remove local file even on upload failure to prevent disk space accumulation
            if chunk_path.exists():
                chunk_path.unlink()
                log.info(f"Removed local chunk {chunk_filename} after upload failure")

    def _move_chunk_locally(
        self,
        chunk_path: Path,
        parent_dir: str,
        original_file_name: str,
        chunk_filename: str,
    ) -> None:
        """Move a chunk to the local directory structure."""
        # Create subdirectory structure: data/chunks/{parent_dir}/{original_file_name}/
        upload_subdir = self.chunks_dir / parent_dir / original_file_name
        upload_subdir.mkdir(parents=True, exist_ok=True)

        # Move the chunk file
        destination_path = upload_subdir / chunk_filename

        if not destination_path.exists():
            shutil.move(str(chunk_path), str(destination_path))
            log.info(f"Moved chunk {chunk_filename} to {destination_path}")
        else:
            log.warning(f"Chunk {chunk_filename} already exists at {destination_path}")
            # Remove the original if it already exists in destination
            chunk_path.unlink()

    def collect_batch(self) -> list[dict]:
        """Collect chunks for a batch."""
        batch = []

        # Add any pending chunks
        batch.extend(self.pending_chunks)
        self.pending_chunks = []

        # Collect new chunks up to batch size
        while len(batch) < self.batch_size:
            chunk_msg = self.redis_client.pop_job(self.input_queue_name, timeout=1)
            if chunk_msg:
                batch.append(chunk_msg)
            else:
                break  # No more messages available

        return batch

    def start_listening(self):
        """Start listening to the transcription queue."""
        log.info(f"Starting batch transcriber, listening to: {self.input_queue_name}")

        if not self.redis_client.health_check():
            log.error("Redis health check failed")
            return

        while True:
            try:
                # Collect a batch of chunks
                batch = self.collect_batch()

                if batch:
                    log.info(f"Collected batch of {len(batch)} chunks")
                    success = self.process_chunk_batch(batch)

                    if not success:
                        log.error("Batch processing failed, requeueing chunks")
                        # Add failed chunks back to pending
                        self.pending_chunks.extend(batch)
                else:
                    # No chunks available, wait a bit
                    time.sleep(1)

            except KeyboardInterrupt:
                log.info("Shutting down batch transcriber")
                break
            except Exception as e:
                log.error(f"Error in transcriber loop: {e}")
                time.sleep(5)


class WhisperTranscriber:
    """Whisper transcriber with batch processing support."""

    def __init__(
        self,
        checkpoint_path: str,
        device: torch.device,
        inference_batch_size: int = 4,
        dtype: torch.dtype = torch.float16,
    ):
        self.dtype = dtype
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            checkpoint_path,
            torch_dtype=dtype,
            attn_implementation="flash_attention_2",
        )
        self.processor = AutoProcessor.from_pretrained(checkpoint_path, use_fast=True)
        self.processor.feature_extractor.return_attention_mask = True  # enable the mask
        self.device = device
        self.inference_batch_size = inference_batch_size
        self.model.to(self.device)

        # Compile model for better performance
        if torch.cuda.is_available():
            log.info("Compiling model with torch.compile for better performance...")
            self.model = torch.compile(self.model)
            log.info("Model compilation completed")

    @torch.no_grad()
    @torch.inference_mode()
    def batch_transcribe(self, audio_data: list[tuple]) -> list[dict]:
        """Batch transcribe multiple audio files."""
        results = []
        target_sr = 16000

        # Process in smaller batches based on inference_batch_size parameter
        for i in range(0, len(audio_data), self.inference_batch_size):
            mini_batch = audio_data[i : i + self.inference_batch_size]
            valid_audio = [audio for audio, _ in mini_batch if audio is not None]

            if not valid_audio:
                # Add empty results for failed audio files
                for _ in mini_batch:
                    results.append({"text": ""})
                continue

            try:
                # Prepare batch input
                inputs = self.processor(
                    valid_audio,
                    sampling_rate=target_sr,
                    return_tensors="pt",
                    padding=True,
                )
                input_features = inputs.input_features.to(self.device, dtype=self.dtype)

                # Generate transcriptions
                generated_ids = self.model.generate(
                    input_features,
                    do_sample=False,
                    num_beams=1,
                    use_cache=True,
                )

                # Decode transcriptions
                transcriptions = self.processor.batch_decode(
                    generated_ids, skip_special_tokens=False
                )

                # Process results
                valid_idx = 0
                for j, (audio, _) in enumerate(mini_batch):
                    if audio is not None:
                        # Clean transcription
                        clean_text = re.sub(
                            r"<\|.*?\|>", "", transcriptions[valid_idx]
                        ).strip()

                        results.append({"text": clean_text})
                        valid_idx += 1
                    else:
                        results.append({"text": ""})

            except Exception as e:
                log.error(f"Batch transcription failed: {e}")
                # Add empty results for this batch
                for _ in mini_batch:
                    results.append({"text": ""})

        return results


def get_available_gpus():
    """Get list of available GPU IDs."""
    if not torch.cuda.is_available():
        log.warning("CUDA not available, falling back to CPU")
        return []

    gpu_count = torch.cuda.device_count()
    log.info(f"Found {gpu_count} GPU(s)")
    return list(range(gpu_count))


def launch_transcriber_process(gpu_id, args):
    """Launch a single transcriber process for a specific GPU."""
    log.info(f"Starting transcriber process on GPU {gpu_id}")

    transcriber = BatchTranscriber(
        input_queue_name=args.input_queue,
        whisper_checkpoint_path=args.whisper_checkpoint,
        whisper_clean_checkpoint_path=args.whisper_clean_checkpoint,
        gpu_id=gpu_id,
        batch_size=args.batch_size,
        inference_batch_size=args.inference_batch_size,
        max_chunk_duration=args.max_chunk_duration,
        chunks_dir=args.chunks_dir,
        upload_to_gcs=args.upload_to_gcs,
        dtype=args.dtype,
    )
    transcriber.start_listening()


def launch_multi_gpu_processes(args):
    """Launch transcriber processes on available GPUs (up to max_gpus limit)."""
    # Set multiprocessing start method to 'spawn' for CUDA compatibility
    multiprocessing.set_start_method("spawn", force=True)

    gpu_ids = get_available_gpus()

    if not gpu_ids:
        log.info("No GPUs available, launching single CPU process")
        launch_transcriber_process(0, args)
        return

    # Limit GPUs if max_gpus is specified
    if hasattr(args, "max_gpus") and args.max_gpus > 0:
        gpu_ids = gpu_ids[: args.max_gpus]
        log.info(f"Limited to {len(gpu_ids)} GPUs (max_gpus={args.max_gpus})")

    processes = []

    def signal_handler(signum, frame):
        log.info("Received interrupt signal, shutting down all processes...")
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=5)
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        for gpu_id in gpu_ids:
            process = multiprocessing.Process(
                target=launch_transcriber_process,
                args=(gpu_id, args),
                name=f"transcriber-gpu-{gpu_id}",
            )
            process.start()
            processes.append(process)
            log.info(f"Launched process {process.pid} for GPU {gpu_id}")

        # Wait for all processes to complete
        for process in processes:
            process.join()

    except Exception as e:
        log.error(f"Error in multi-GPU launcher: {e}")
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=5)


def main():
    """Main function for batch transcriber."""
    parser = argparse.ArgumentParser(description="Batch audio transcriber")

    parser.add_argument(
        "--input-queue",
        default="transcription_queue",
        help="Input queue name for chunks",
    )
    parser.add_argument(
        "--whisper-checkpoint",
        default="pevers/whisperd-nl",
        help="Whisper model checkpoint path",
    )
    parser.add_argument(
        "--whisper-clean-checkpoint",
        default="pevers/whisper-clean-nl",
        help="Clean whisper model checkpoint path",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="GPU ID for processing",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of chunks to collect before processing",
    )
    parser.add_argument(
        "--inference-batch-size",
        type=int,
        default=4,
        help="Batch size for model inference",
    )
    parser.add_argument(
        "--max-chunk-duration",
        type=float,
        default=30.0,
        help="Maximum chunk duration in seconds",
    )
    parser.add_argument(
        "--chunks-dir",
        default="data/chunks",
        help="Directory for storing processed chunks for GCS upload (only used when --upload-to-gcs is False)",
    )
    parser.add_argument(
        "--upload-to-gcs",
        action="store_true",
        help="Upload processed chunks directly to Google Cloud Storage instead of storing locally",
    )
    parser.add_argument(
        "--multi-gpu",
        action="store_true",
        help="Launch one process per available GPU automatically",
    )
    parser.add_argument(
        "--max-gpus",
        type=int,
        default=0,
        help="Maximum number of GPUs to use (0 = use all available)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Data type for model inference (default: float16)",
    )

    args = parser.parse_args()

    # Convert dtype string to torch dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    args.dtype = dtype_map[args.dtype]

    if args.multi_gpu:
        launch_multi_gpu_processes(args)
    else:
        transcriber = BatchTranscriber(
            input_queue_name=args.input_queue,
            whisper_checkpoint_path=args.whisper_checkpoint,
            whisper_clean_checkpoint_path=args.whisper_clean_checkpoint,
            gpu_id=args.gpu_id,
            batch_size=args.batch_size,
            inference_batch_size=args.inference_batch_size,
            max_chunk_duration=args.max_chunk_duration,
            chunks_dir=args.chunks_dir,
            upload_to_gcs=args.upload_to_gcs,
            dtype=args.dtype,
        )
        transcriber.start_listening()


if __name__ == "__main__":
    main()
