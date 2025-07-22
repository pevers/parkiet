import argparse
import logging
import time
import torch
import re
from pathlib import Path
from typing import Any
from dotenv import load_dotenv
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import librosa

load_dotenv()
from parkiet.audioprep.schemas import AudioChunk, ProcessedAudioChunk
from parkiet.database.audio_store import AudioStore
from parkiet.database.redis_connection import get_redis_connection

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
    ):
        self.input_queue_name = input_queue_name
        self.batch_size = batch_size
        self.inference_batch_size = inference_batch_size
        self.max_chunk_duration = max_chunk_duration

        # Initialize services
        self.redis_client = get_redis_connection()
        self.audio_store = AudioStore()

        # Initialize device configuration
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{gpu_id}")
            torch.cuda.set_device(gpu_id)
        else:
            self.device = torch.device("cpu")

        # Load models at initialization and keep them loaded
        log.info("Loading transcription models...")
        self.transcriber = WhisperTranscriber(
            whisper_checkpoint_path, self.device, self.inference_batch_size
        )
        self.clean_transcriber = WhisperTranscriber(
            whisper_clean_checkpoint_path, self.device, self.inference_batch_size
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

            # Extract audio paths for batch processing
            audio_paths = []
            for chunk_msg in valid_chunks:
                chunk_path = chunk_msg["chunk_path"]
                if Path(chunk_path).exists():
                    audio_paths.append(chunk_path)
                else:
                    log.warning(f"Chunk file not found: {chunk_path}")

            if not audio_paths:
                log.warning("No valid audio files found")
                return False

            # Batch transcribe with both models
            log.info(
                f"Running batch transcription with whisperd-nl model (inference batch size: {self.inference_batch_size})"
            )
            whisperd_results = self.transcriber.batch_transcribe_with_confidence(
                audio_paths
            )

            log.info(
                f"Running batch transcription with whisper-clean-nl model (inference batch size: {self.inference_batch_size})"
            )
            clean_results = self.clean_transcriber.batch_transcribe_with_confidence(
                audio_paths
            )

            # Process results and store in database
            processed_chunks = []
            for i, chunk_msg in enumerate(valid_chunks):
                if i < len(whisperd_results) and i < len(clean_results):
                    chunk_data = AudioChunk(**chunk_msg["chunk_data"])

                    processed_chunk = ProcessedAudioChunk(
                        audio_chunk=chunk_data,
                        transcription=whisperd_results[i]["text"],
                        transcription_conf=whisperd_results[i]["confidence"],
                        transcription_clean=clean_results[i]["text"],
                        transcription_clean_conf=clean_results[i]["confidence"],
                    )

                    processed_chunks.append(processed_chunk)

                    log.info(f"Processed chunk {chunk_data.file_path}")
                    log.info(f"Whisperd: {whisperd_results[i]['text']}...")
                    log.info(f"Clean: {clean_results[i]['text']}...")

            # Store processed chunks in database
            if processed_chunks:
                try:
                    # Group by job_id for database storage
                    jobs_data = {}
                    for i, chunk_msg in enumerate(valid_chunks):
                        if i < len(processed_chunks):
                            job_id = chunk_msg["job_id"]
                            if job_id not in jobs_data:
                                jobs_data[job_id] = {
                                    "gcs_audio_path": chunk_msg["gcs_audio_path"],
                                    "speaker_embeddings": chunk_msg[
                                        "speaker_embeddings"
                                    ],
                                    "chunks": [],
                                }
                            jobs_data[job_id]["chunks"].append(processed_chunks[i])

                    # Store each job's data
                    for job_id, job_data in jobs_data.items():
                        # Create minimal ProcessedAudioFile for storage
                        from parkiet.audioprep.schemas import ProcessedAudioFile

                        processed_file = ProcessedAudioFile(
                            source_file=job_data["gcs_audio_path"],
                            output_directory="",  # Not used in batch mode
                            gcs_audio_path=job_data["gcs_audio_path"],
                            audio_duration_sec=0.0,  # Will be calculated if needed
                            chunks=job_data["chunks"],
                            success=True,
                        )

                        audio_file_id = self.audio_store.store_processed_file(
                            processed_file, job_data["speaker_embeddings"]
                        )
                        log.info(
                            f"Stored job {job_id} in database with ID: {audio_file_id}"
                        )

                except Exception as e:
                    log.error(f"Failed to store processed chunks in database: {e}")

            log.info(f"Successfully processed batch of {len(processed_chunks)} chunks")
            return True

        except Exception as e:
            log.error(f"Error processing chunk batch: {e}")
            return False

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
        self, checkpoint_path: str, device: torch.device, inference_batch_size: int = 4
    ):
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            checkpoint_path, low_cpu_mem_usage=True, use_safetensors=True
        )
        self.processor = AutoProcessor.from_pretrained(checkpoint_path)
        self.device = device
        self.inference_batch_size = inference_batch_size
        self.model.to(self.device)

        # Compile model for better performance (requires PyTorch 2.0+)
        if hasattr(torch, "compile") and torch.cuda.is_available():
            log.info("Compiling model with torch.compile for better performance...")
            self.model = torch.compile(self.model)
            log.info("Model compilation completed")

    @torch.no_grad()
    @torch.inference_mode()
    def batch_transcribe_with_confidence(self, audio_paths: list[str]) -> list[dict]:
        """Batch transcribe multiple audio files with confidence scores."""
        results = []
        target_sr = 16000

        # Load all audio files
        audio_batch = []
        for audio_path in audio_paths:
            try:
                audio, sr = librosa.load(audio_path, sr=target_sr)
                if sr != target_sr:
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
                audio_batch.append(audio)
            except Exception as e:
                log.error(f"Failed to load audio {audio_path}: {e}")
                audio_batch.append(None)

        # Process in smaller batches based on inference_batch_size parameter
        for i in range(0, len(audio_batch), self.inference_batch_size):
            mini_batch = audio_batch[i : i + self.inference_batch_size]
            valid_audio = [audio for audio in mini_batch if audio is not None]

            if not valid_audio:
                # Add empty results for failed audio files
                for _ in mini_batch:
                    results.append({"text": "", "confidence": 0.0})
                continue

            try:
                # Prepare batch input
                inputs = self.processor(
                    valid_audio,
                    sampling_rate=target_sr,
                    return_tensors="pt",
                    padding=True,
                )
                input_features = inputs.input_features.to(self.device)

                # Generate transcriptions
                generation_result = self.model.generate(
                    input_features,
                    do_sample=False,
                    output_scores=True,
                    num_beams=3,  # Reduced beams for batch processing
                    return_dict_in_generate=True,
                )

                # Calculate confidence scores
                logp = self.model.compute_transition_scores(
                    generation_result.sequences,
                    generation_result.scores,
                    normalize_logits=True,
                )

                # Decode transcriptions
                transcriptions = self.processor.batch_decode(
                    generation_result.sequences, skip_special_tokens=False
                )

                # Process results
                valid_idx = 0
                for j, audio in enumerate(mini_batch):
                    if audio is not None:
                        # Clean transcription
                        clean_text = re.sub(
                            r"<\|.*?\|>", "", transcriptions[valid_idx]
                        ).strip()

                        # Calculate confidence for this sequence
                        seq_logp = logp[valid_idx]
                        seq_conf = torch.exp(seq_logp.mean())

                        results.append(
                            {"text": clean_text, "confidence": float(seq_conf)}
                        )
                        valid_idx += 1
                    else:
                        results.append({"text": "", "confidence": 0.0})

            except Exception as e:
                log.error(f"Batch transcription failed: {e}")
                # Add empty results for this batch
                for _ in mini_batch:
                    results.append({"text": "", "confidence": 0.0})

        return results


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

    args = parser.parse_args()

    transcriber = BatchTranscriber(
        input_queue_name=args.input_queue,
        whisper_checkpoint_path=args.whisper_checkpoint,
        whisper_clean_checkpoint_path=args.whisper_clean_checkpoint,
        gpu_id=args.gpu_id,
        batch_size=args.batch_size,
        inference_batch_size=args.inference_batch_size,
        max_chunk_duration=args.max_chunk_duration,
    )
    transcriber.start_listening()


if __name__ == "__main__":
    main()
