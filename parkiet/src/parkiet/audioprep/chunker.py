import json
import argparse
import logging
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
import time
import torch
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
from parkiet.audioprep.arrow_writer import convert_to_arrow_table
from parkiet.dia.model import Dia
from ulid import ULID

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)


def create_chunks(
    speaker_events: list[SpeakerEvent],
    original_audio_path: Path,
    output_dir: Path,
    window_size_sec: float = 30.0,
    skip_start_sec: float = 120.0,  # 2 minutes
    skip_end_sec: float = 180.0,  # 3 minutes
) -> tuple[list[AudioChunk], dict[str, float], float]:
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
        Tuple of (chunks, processing_window, audio_duration)
        - chunks: List of AudioChunk objects
        - processing_window: Dict with 'start' and 'end' keys in seconds
        - audio_duration: Total audio duration in seconds
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
        processing_window = {"start": actual_start, "end": actual_end}
        return [], processing_window, audio_duration

    chunks = []
    current_time = actual_start

    while current_time < actual_end:
        window_end = min(current_time + window_size_sec, actual_end)

        # Find events in this window
        window_events = []
        for event in events_in_window:
            if event.start < window_end and event.end > current_time:
                # Clip event to window
                clipped_start = max(event.start, current_time)
                clipped_end = min(event.end, window_end)
                window_events.append(
                    SpeakerEvent(
                        start=clipped_start, end=clipped_end, speaker=event.speaker
                    )
                )

        if window_events:
            chunk_id = str(ULID())
            chunk_filename = f"{chunk_id}.mp3"
            chunk_path = output_dir / chunk_filename

            extract_audio_segment(
                original_audio_path, current_time, window_end, chunk_path
            )

            chunk = AudioChunk(
                start=current_time * 1000,
                end=window_end * 1000,
                file_path=chunk_filename,
                speaker_events=window_events,
            )

            chunks.append(chunk)
            log.debug(
                f"Created chunk {len(chunks)}: {current_time:.1f}s-{window_end:.1f}s"
            )

        # Move window forward (could be overlapping or non-overlapping based on requirements)
        current_time += window_size_sec

    return chunks, audio_duration


def process_single_audio_file(
    audio_file_path: Path,
    target_folder: Path,
    whisper_checkpoint_path: str,
    window_size_sec: float = 30.0,
    skip_start_sec: float = 120.0,
    skip_end_sec: float = 180.0,
) -> ProcessedAudioFile:
    """
    Process a single audio file.

    Args:
        audio_file_path: Path to the audio file
        target_folder: Target folder for output
        whisper_checkpoint_path: Path to the Whisper checkpoint
        window_size_sec: Size of sliding window in seconds
        skip_start_sec: Time to skip from start
        skip_end_sec: Time to skip from end

    Returns:
        ProcessedAudioFile object
    """
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    speaker_extractor = SpeakerExtractor(device)
    transcriber = Transcriber(whisper_checkpoint_path, device)

    try:
        # Create output directory based on audio file name (without extension)
        output_dir_name = audio_file_path.stem
        output_dir_path = target_folder / output_dir_name
        output_dir_path.mkdir(parents=True, exist_ok=True)

        # Check if this file has already been processed
        processed_file_json_path = output_dir_path / "processed_file.json"
        if processed_file_json_path.exists():
            log.info(
                f"Skipping {audio_file_path.name} - already processed (found {processed_file_json_path})"
            )
            with open(processed_file_json_path, "r") as f:
                processed_data = json.load(f)
            return ProcessedAudioFile(**processed_data)

        log.info(f"Processing: {audio_file_path.name}")
        log.info(f"Output directory: {output_dir_path}")
        log.info("Extracting speaker events with pyannote...")
        speaker_events = speaker_extractor.extract_speaker_events(audio_file_path)
        log.info(f"Found {len(speaker_events)} speaker events")

        if not speaker_events:
            log.warning("No speaker events found, skipping chunk creation")
            return ProcessedAudioFile(
                source_file=audio_file_path.as_posix(),
                output_directory=output_dir_path.as_posix(),
                audio_duration_sec=get_audio_duration(audio_file_path),
                chunks=[],
                processing_window={"start": 0.0, "end": 0.0},
                success=False,
            )

        log.info("Creating audio chunks...")
        chunks, audio_duration = create_chunks(
            speaker_events,
            audio_file_path,
            output_dir_path,
            window_size_sec,
            skip_start_sec,
            skip_end_sec,
        )

        log.info(f"Creating transcription for {len(chunks)} chunks")
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            chunk_full_path = output_dir_path / chunk.file_path
            transcription = transcriber.transcribe(chunk_full_path)
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
            output_directory=output_dir_path.as_posix(),
            audio_duration_sec=audio_duration,
            chunks=processed_chunks,
            success=True,
        )
        processed_file_json_path = output_dir_path / "processed_file.json"
        with open(processed_file_json_path, "w") as f:
            json.dump(processed_file.model_dump(), f, indent=2)

        log.info(f"Completed: {len(chunks)} chunks in {time.time() - start_time:.1f}s")
        return processed_file

    except Exception as e:
        log.error(f"Error processing {audio_file_path.name}: {str(e)}")
        return ProcessedAudioFile(
            source_file=str(audio_file_path),
            output_directory=str(output_dir_path),
            audio_duration_sec=get_audio_duration(audio_file_path),
            chunks=[],
            processing_window={"start": 0.0, "end": 0.0},
            success=False,
        )


def preprocess_audio_batch(
    source_folder: str,
    target_folder: str,
    whisper_checkpoint_path: str,
    max_workers: Optional[int] = None,
    window_size_sec: float = 30.0,
    skip_start_sec: float = 120.0,
    skip_end_sec: float = 180.0,
) -> None:
    """
    Main function to preprocess all audio files in a source folder.

    Args:
        source_folder: Path to the source folder containing audio files
        target_folder: Path to the target folder for output
        whisper_checkpoint_path: Path to the Whisper checkpoint
        max_workers: Maximum number of worker threads (defaults to CPU count)
        window_size_sec: Size of sliding window in seconds
        skip_start_sec: Time to skip from start (will find natural break after this)
        skip_end_sec: Time to skip from end (will find natural break before this)
    """
    dia = Dia.from_local(
        config_path="config.json", checkpoint_path="weights/dia-v0_1.pth"
    )
    source_path = Path(source_folder)
    target_path = Path(target_folder)

    if not source_path.exists():
        raise FileNotFoundError(f"Source folder not found: {source_path}")

    # Create target directory if it doesn't exist
    target_path.mkdir(parents=True, exist_ok=True)
    audio_files = find_audio_files(source_path)

    if not audio_files:
        log.warning(f"No audio files found in: {source_path}")
        return

    log.info(f"Found {len(audio_files)} audio files in: {source_path}")
    log.info(f"Target folder: {target_path}")
    log.info(
        f"Window size: {window_size_sec}s, Skip start: {skip_start_sec}s, Skip end: {skip_end_sec}s"
    )

    if max_workers is None:
        max_workers = min(multiprocessing.cpu_count(), len(audio_files))

    log.info(f"Using {max_workers} worker threads")
    results = []

    def process_with_params(audio_file):
        return process_single_audio_file(
            audio_file,
            target_path,
            whisper_checkpoint_path,
            window_size_sec,
            skip_start_sec,
            skip_end_sec,
        )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(process_with_params, audio_file): audio_file
            for audio_file in audio_files
        }
        for future in as_completed(future_to_file):
            result = future.result()
            results.append(result)

    # Generate overview JSON
    overview = {
        "source_folder": str(source_path),
        "target_folder": str(target_path),
        "total_files_processed": len(results),
        "successful_files": len([r for r in results if r]),
        "failed_files": len([r for r in results if not r.success]),
        "total_chunks": sum(len(r.chunks) for r in results if r.success),
        "processing_params": {
            "window_size_sec": window_size_sec,
            "skip_start_sec": skip_start_sec,
            "skip_end_sec": skip_end_sec,
            "max_workers": max_workers,
        },
        "results": [result.model_dump() for result in results],
    }

    overview_path = target_path / "processing_overview.json"
    with open(overview_path, "w") as f:
        json.dump(overview, f, indent=2)

    log.info("\n=== Processing Complete ===")
    log.info(f"Total files: {len(results)}")
    log.info(f"Successful: {overview['successful_files']}")
    log.info(f"Failed: {overview['failed_files']}")
    log.info(f"Total chunks created: {overview['total_chunks']}")
    log.info(f"Overview saved to: {overview_path}")

    # Print failed files if any
    failed_results = [r for r in results if not r.success]
    if failed_results:
        log.warning("\nFailed files:")
        for result in failed_results:
            log.warning(f"- {Path(result.source_file).name}")

    log.info("Converting processed files to Apache Arrow format...")
    convert_to_arrow_table(dia, target_path)


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Create audio chunks from source files with speaker diarization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s /path/to/source /path/to/target
  %(prog)s /path/to/source /path/to/target --window-size 45 --workers 4
  %(prog)s /path/to/source /path/to/target --skip-start 60 --skip-end 120
        """,
    )

    parser.add_argument(
        "source_folder", help="Path to source folder containing audio files"
    )

    parser.add_argument("target_folder", help="Path to target folder for output chunks")

    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=None,
        help="Maximum number of worker threads (default: min(CPU_count, file_count))",
    )
    parser.add_argument(
        "--whisper-checkpoint-path",
        "-c",
        type=str,
        default="openai/whisper-large-v3",
        help="Path to Whisper checkpoint (default: openai/whisper-large-v3)",
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
    preprocess_audio_batch(
        source_folder=args.source_folder,
        target_folder=args.target_folder,
        max_workers=args.workers,
        window_size_sec=args.window_size,
        skip_start_sec=args.skip_start,
        skip_end_sec=args.skip_end,
        whisper_checkpoint_path=args.whisper_checkpoint_path,
    )


if __name__ == "__main__":
    main()
