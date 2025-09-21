import argparse
import logging
import time
from pathlib import Path
from typing import Any
from dataclasses import dataclass
from google.cloud.exceptions import NotFound
import pyarrow as pa
import pyarrow.parquet as pq
from parkiet.database.audio_store import AudioStore
from parkiet.dia.config import DiaConfig
from parkiet.storage.gcs_client import GCSClient
from parkiet.dia.model import ComputeDtype, Dia
import tempfile
from multiprocessing import Pool

log = logging.getLogger(__name__)

CHUNKS_PER_SHARD = 40000


@dataclass
class SpeakerStat:
    duration_sec: float
    sample_prob: float
    cb_weight: float  # for loss


def _get_shard_output_path(
    output_path: Path, shard_idx: int, total_shards: int
) -> Path:
    """Generate the output path for a specific shard."""
    if total_shards == 1:
        return output_path
    else:
        stem = output_path.stem
        suffix = output_path.suffix
        return output_path.parent / f"{stem}_shard_{shard_idx + 1:03d}{suffix}"


def _get_shards_to_process(output_path: Path, total_shards: int) -> list[int]:
    """Determine which shards need to be processed by checking existing files."""
    shards_to_process = []
    skipped_shards = []

    for shard_idx in range(total_shards):
        shard_output_path = _get_shard_output_path(output_path, shard_idx, total_shards)

        # Check if local file exists
        if shard_output_path.exists():
            log.info(
                f"Shard {shard_idx + 1}/{total_shards} already exists at {shard_output_path}, skipping"
            )
            skipped_shards.append(shard_idx + 1)
            continue

        # If we get here, the shard needs to be processed
        shards_to_process.append(shard_idx)

    if skipped_shards:
        log.info(
            f"Skipping {len(skipped_shards)} already completed shards: {skipped_shards}"
        )

    return shards_to_process


def _process_shard_worker(shard_info: tuple) -> str:
    """Worker function to process a single shard in a separate process."""
    (
        shard_chunks,
        shard_idx,
        total_shards,
        output_path,
        data_directory,
        dia_config_path,
    ) = shard_info

    # Initialize components in worker process
    audio_store = AudioStore()
    speaker_stats = audio_store.calculate_speaker_stats()

    dia_config = DiaConfig.load(dia_config_path)
    dia = Dia(config=dia_config, compute_dtype=ComputeDtype.BFLOAT16, load_dac=True)
    dia._load_dac_model()
    gcs_client = GCSClient()

    # Generate shard output path
    shard_output_path = _get_shard_output_path(output_path, shard_idx, total_shards)

    # Process chunks for this shard
    shard_data = []
    total_chunks = len(shard_chunks)
    progress_interval = max(1, total_chunks // 1000)  # Log every 0.1%

    for i, chunk in enumerate(shard_chunks):
        if (i + 1) % progress_interval == 0 or i == total_chunks - 1:
            progress_percent = ((i + 1) / total_chunks) * 100
            log.info(
                f"Processing chunk {chunk['chunk_id']} - Progress: {progress_percent:.1f}% ({i + 1}/{total_chunks})"
            )

        chunk_data = _extract_chunk_data_from_db(
            dia, chunk, speaker_stats, audio_store, data_directory, gcs_client
        )
        if chunk_data is not None:
            shard_data.append(chunk_data)

    _write_shard_to_parquet(shard_data, shard_output_path)

    return f"Shard {shard_idx + 1}/{total_shards}: {len(shard_chunks)} chunks -> {shard_output_path}"


def convert_to_arrow_table(
    dia_config_path: str,
    output_path: Path,
    chunk_limit: int | None = None,
    data_directory: Path | None = None,
    num_workers: int = 1,
) -> None:
    """Convert processed audio chunks from database to Apache Arrow format.

    Args:
        dia_config_path: Path to Dia config file
        output_path: Path where to save the parquet file(s)
        chunk_limit: Optional limit on number of chunks to process
        data_directory: Local directory containing audio chunks, if None uses GCS; if provided but file not found, will fallback to GCS
        num_workers: Number of worker processes to use
    """
    audio_store = AudioStore()

    # Get audio chunks from database
    log.info(f"Fetching audio chunks from database (limit: {chunk_limit})...")
    chunks = audio_store.get_all_audio_chunks(limit=chunk_limit)

    if not chunks:
        log.warning("No audio chunks found")
        return

    log.info(f"Found {len(chunks)} audio chunks to process")

    # Compression with DAC is a lot, so we should bundle a ton of them
    chunks_per_shard = CHUNKS_PER_SHARD
    total_shards = (len(chunks) + chunks_per_shard - 1) // chunks_per_shard

    log.info(
        f"Creating {total_shards} shards with ~{chunks_per_shard} chunks each using {num_workers} workers"
    )

    # Check which shards need to be processed
    shards_to_process = _get_shards_to_process(output_path, total_shards)

    if not shards_to_process:
        log.info("All shards already exist! Nothing to process.")
        return

    log.info(f"Need to process {len(shards_to_process)} out of {total_shards} shards")
    log.info(f"Shards to process: {[idx + 1 for idx in shards_to_process]}")

    # Prepare shard info for workers (only for shards that need processing)
    shard_infos = []
    for shard_idx in shards_to_process:
        start_idx = shard_idx * chunks_per_shard
        end_idx = min(start_idx + chunks_per_shard, len(chunks))
        shard_chunks = chunks[start_idx:end_idx]

        shard_info = (
            shard_chunks,
            shard_idx,
            total_shards,
            output_path,
            data_directory,
            dia_config_path,
        )
        shard_infos.append(shard_info)

    # Process shards using multiprocessing
    if num_workers == 1:
        # Single process mode
        for shard_info in shard_infos:
            start_time = time.time()
            result = _process_shard_worker(shard_info)
            shard_time = time.time() - start_time
            log.info(f"{result} (took {shard_time:.2f}s)")
    else:
        # Multi-process mode
        with Pool(num_workers) as pool:
            results = pool.map(_process_shard_worker, shard_infos)
            for result in results:
                log.info(result)

    # Final summary
    processed_count = len(shards_to_process)
    skipped_count = total_shards - processed_count

    if skipped_count > 0:
        log.info("=== CONVERSION SUMMARY ===")
        log.info(f"Total shards: {total_shards}")
        log.info(f"Processed: {processed_count}")
        log.info(f"Skipped (already existed): {skipped_count}")
        log.info(f"Output location: {output_path}")
        log.info("==========================")
    else:
        log.info(f"Conversion complete! All {total_shards} shards processed")


def _write_shard_to_parquet(chunk_data: list[dict], output_path: Path) -> None:
    """Write chunk data to parquet file and optionally upload to GCS."""
    log.info(f"Creating Arrow table with {len(chunk_data)} chunks...")

    # Create Arrow table
    column_data = {}
    schema = _define_arrow_schema()

    for field in schema:
        column_data[field.name] = []

    # Populate column data from row data
    for row in chunk_data:
        for column_name in column_data.keys():
            column_data[column_name].append(row[column_name])

    table = pa.table(column_data, schema=schema)

    # Write file with zstd compression
    pq.write_table(table, output_path, compression="zstd")
    log.info(f"Shard saved to: {output_path} ({table.shape[0]} rows)")


def _define_arrow_schema() -> pa.Schema:
    """Define the Arrow schema for the dataset."""
    return pa.schema(
        [
            ("source_file", pa.string()),
            ("chunk_id", pa.string()),
            ("start_ms", pa.float64()),
            ("end_ms", pa.float64()),
            ("duration_ms", pa.float64()),
            ("transcription", pa.string()),
            ("transcription_clean", pa.string()),
            ("file_path", pa.string()),
            ("chunk_owner", pa.int64()),  # Speaker ID who speaks the most in this chunk
            ("sample_prob", pa.float64()),  # Sampling probability for this chunk
            ("cb_weight", pa.float64()),  # Class-balanced weight for this chunk
            ("encoded_audio_shape", pa.list_(pa.int64())),
            ("encoded_audio", pa.list_(pa.int64())),
        ]
    )


def _extract_chunk_data_from_db(
    dia: Dia,
    chunk_data: dict,
    speaker_stats: dict[int, SpeakerStat],
    audio_store: AudioStore,
    data_directory: Path | None,
    gcs_client: GCSClient | None,
) -> dict[str, Any] | None:
    """Extract chunk data from database record for Arrow table.

    Audio data is loaded from data_directory if provided, with automatic fallback to GCS
    if the file is not found locally.
    """
    chunk_id = chunk_data["chunk_id"]
    chunk_file_path = chunk_data["chunk_file_path"]

    # Get chunk owner (speaker who speaks the most in this chunk)
    chunk_owner = audio_store.get_chunk_owner(chunk_id)

    # Get sampling probability and class-balanced weight from chunk owner
    sample_prob = 0.0
    cb_weight = 1.0
    if chunk_owner and chunk_owner in speaker_stats:
        sample_prob = speaker_stats[chunk_owner].sample_prob
        cb_weight = speaker_stats[chunk_owner].cb_weight

    # Load audio data
    encoded_audio = _load_audio_data(chunk_file_path, data_directory, gcs_client, dia)
    if encoded_audio is None:
        return None

    encoded_audio_shape = list(encoded_audio.shape)
    encoded_audio_flat = encoded_audio.flatten().tolist()

    return {
        "source_file": chunk_data["original_file_path"],
        "chunk_id": str(chunk_id),
        "start_ms": float(chunk_data["start_time_ms"]),
        "end_ms": float(chunk_data["end_time_ms"]),
        "duration_ms": float(chunk_data["end_time_ms"] - chunk_data["start_time_ms"]),
        "transcription": chunk_data["transcription"] or "",
        "transcription_clean": chunk_data["transcription_clean"] or "",
        "file_path": chunk_file_path,
        "chunk_owner": chunk_owner
        if chunk_owner is not None
        else -1,  # Use -1 for no owner
        "sample_prob": sample_prob,
        "cb_weight": cb_weight,
        "encoded_audio_shape": encoded_audio_shape,
        "encoded_audio": encoded_audio_flat,
    }


def _load_audio_data(
    chunk_file_path: str,
    data_directory: Path | None,
    gcs_client: GCSClient | None,
    dia: Dia,
) -> Any:
    """Load audio data from local directory or GCS.

    If data_directory is provided but the file is not found locally,
    automatically falls back to GCS if available.
    """
    if data_directory:
        # Try to load from local directory first
        local_path = data_directory / chunk_file_path
        if local_path.exists():
            return dia.load_audio(local_path)
        else:
            # File not found locally, try GCS fallback if available
            if gcs_client:
                log.info(f"Audio chunk not found at {local_path}, falling back to GCS")
                try:
                    # Load from GCS as fallback
                    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp_file:
                        # Download from GCS to temporary file
                        blob = gcs_client.bucket.blob(f"chunks/{chunk_file_path}")
                        blob.download_to_filename(tmp_file.name)

                        # Load audio from temporary file
                        encoded_audio = dia.load_audio(Path(tmp_file.name))

                        return encoded_audio
                except NotFound:
                    log.warning(f"Could not find {chunk_file_path}, skipping")
                    return None
            else:
                # No GCS fallback available
                raise FileNotFoundError(
                    f"Audio chunk not found at {local_path} and no GCS fallback available"
                )

    elif gcs_client:
        try:
            # Load from GCS directly
            with tempfile.NamedTemporaryFile(suffix=".wav") as tmp_file:
                # Download from GCS to temporary file
                blob = gcs_client.bucket.blob(f"chunks/{chunk_file_path}")
                blob.download_to_filename(tmp_file.name)

                # Load audio from temporary file
                encoded_audio = dia.load_audio(Path(tmp_file.name))

                return encoded_audio
        except NotFound:
            log.warning(f"Could not find {chunk_file_path}, skipping")
            return None

    else:
        raise ValueError("Either data_directory or gcs_client must be provided")


def main():
    """CLI entry point for the arrow writer."""
    parser = argparse.ArgumentParser(
        description="Convert audio chunks from database to Apache Arrow format"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("chunks_dataset.parquet"),
        help="Output path for the parquet file (default: chunks_dataset.parquet)",
    )
    parser.add_argument(
        "--limit", "-l", type=int, help="Limit number of chunks to process"
    )
    parser.add_argument(
        "--data-directory",
        "-d",
        type=Path,
        help="Local directory containing audio chunk files (if not provided, will fetch from GCS; if provided but file not found, will fallback to GCS)",
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=1,
        help="Number of worker processes to use (default: 1)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Set up logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Convert to arrow table
    convert_to_arrow_table(
        dia_config_path="config.json",
        output_path=args.output,
        chunk_limit=args.limit,
        data_directory=args.data_directory,
        num_workers=args.workers,
    )


if __name__ == "__main__":
    main()
