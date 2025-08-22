import argparse
import logging
import time
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from parkiet.database.audio_store import AudioStore
from parkiet.dia.config import DiaConfig
from parkiet.storage.gcs_client import GCSClient
from parkiet.dia.model import ComputeDtype, Dia
import tempfile
from multiprocessing import Pool
import json

log = logging.getLogger(__name__)


def save_chunks_to_file(chunks: list[dict], output_file: Path) -> None:
    """Save chunks to a JSON file."""
    with open(output_file, 'w') as f:
        json.dump(chunks, f, indent=2)
    log.info(f"Saved {len(chunks)} chunks to {output_file}")


def load_chunks_from_file(input_file: Path) -> list[dict]:
    """Load chunks from a JSON file."""
    with open(input_file, 'r') as f:
        chunks = json.load(f)
    log.info(f"Loaded {len(chunks)} chunks from {input_file}")
    return chunks


def split_chunks_file(input_file: Path, output_dir: Path = None) -> tuple[Path, Path]:
    """Split chunks file into two equal parts (set1 and set2)."""
    chunks = load_chunks_from_file(input_file)
    
    if output_dir is None:
        output_dir = input_file.parent
    
    mid_point = len(chunks) // 2
    set1_chunks = chunks[:mid_point]
    set2_chunks = chunks[mid_point:]
    
    set1_file = output_dir / f"{input_file.stem}_set1.json"
    set2_file = output_dir / f"{input_file.stem}_set2.json"
    
    save_chunks_to_file(set1_chunks, set1_file)
    save_chunks_to_file(set2_chunks, set2_file)
    
    log.info(f"Split {len(chunks)} chunks into {len(set1_chunks)} (set1) and {len(set2_chunks)} (set2)")
    
    return set1_file, set2_file


def _process_shard_worker(shard_info: tuple) -> str:
    """Worker function to process a single shard in a separate process."""
    (
        shard_chunks,
        shard_idx,
        total_shards,
        output_path,
        data_directory,
        dia_config_path,
        gcs_upload_path,
    ) = shard_info

    # Initialize components in worker process
    audio_store = AudioStore()
    speaker_weights = audio_store.calculate_speaker_weights()

    dia_config = DiaConfig.load(dia_config_path)
    dia = Dia(config=dia_config, compute_dtype=ComputeDtype.BFLOAT16, load_dac=True)
    dia._load_dac_model()

    gcs_client = None
    if data_directory is None:
        gcs_client = GCSClient()

    # Generate shard output path
    if total_shards == 1:
        shard_output_path = output_path
    else:
        stem = output_path.stem
        suffix = output_path.suffix
        shard_output_path = (
            output_path.parent / f"{stem}_shard_{shard_idx + 1:03d}{suffix}"
        )

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
            dia, chunk, speaker_weights, audio_store, data_directory, gcs_client
        )
        shard_data.append(chunk_data)

    _write_shard_to_parquet(shard_data, shard_output_path, gcs_upload_path)

    return f"Shard {shard_idx + 1}/{total_shards}: {len(shard_chunks)} chunks -> {shard_output_path}"


def convert_to_arrow_table(
    dia_config_path: str,
    output_path: Path,
    chunk_limit: int | None = None,
    data_directory: Path | None = None,
    num_workers: int = 1,
    gcs_upload_path: str | None = None,
    chunks_file: Path | None = None,
) -> None:
    """Convert processed audio chunks from database to Apache Arrow format.

    Args:
        dia_config_path: Path to Dia config file
        output_path: Path where to save the parquet file(s)
        chunk_limit: Optional limit on number of chunks to process
        data_directory: Local directory containing audio chunks, if None uses GCS
        num_workers: Number of worker processes to use
        gcs_upload_path: Optional GCS path to upload shards to, removes local files after upload
        chunks_file: Optional path to JSON file containing chunks data (if provided, skips database query)
    """
    audio_store = AudioStore()

    # Get audio chunks from file or database
    if chunks_file:
        log.info(f"Loading audio chunks from file: {chunks_file}")
        chunks = load_chunks_from_file(chunks_file)
        if chunk_limit:
            chunks = chunks[:chunk_limit]
            log.info(f"Limited chunks to {len(chunks)} due to --limit parameter")
    else:
        log.info(f"Fetching audio chunks from database (limit: {chunk_limit})...")
        chunks = audio_store.get_all_audio_chunks(limit=chunk_limit)

    if not chunks:
        log.warning("No audio chunks found")
        return

    log.info(f"Found {len(chunks)} audio chunks to process")

    # Compression with DAC is a lot, so we should bundle a ton of them
    chunks_per_shard = 40000
    total_shards = (len(chunks) + chunks_per_shard - 1) // chunks_per_shard

    log.info(
        f"Creating {total_shards} shards with ~{chunks_per_shard} chunks each using {num_workers} workers"
    )

    # Prepare shard info for workers
    shard_infos = []
    for shard_idx in range(total_shards):
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
            gcs_upload_path,
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

    log.info(f"Conversion complete! Created {total_shards} shard(s)")


def _write_shard_to_parquet(
    chunk_data: list[dict], output_path: Path, gcs_upload_path: str | None = None
) -> None:
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

    # Upload to GCS if path provided
    if gcs_upload_path:
        # Parse GCS URL to extract bucket and path
        if gcs_upload_path.startswith("gs://"):
            # Extract bucket name and path from gs://bucket/path format
            parts = gcs_upload_path[5:].split(
                "/", 1
            )  # Remove "gs://" and split on first "/"
            if len(parts) >= 2:
                upload_bucket = parts[0]
                upload_path = parts[1]
            else:
                upload_bucket = parts[0]
                upload_path = ""
        else:
            # Assume it's just a path in the default bucket
            upload_bucket = None
            upload_path = gcs_upload_path

        # Create GCS client with the specified bucket (if different from default)
        gcs_client = (
            GCSClient(bucket_name=upload_bucket) if upload_bucket else GCSClient()
        )

        # Construct the full GCS path
        if upload_path:
            gcs_path = upload_path.rstrip("/") + "/" + output_path.name
        else:
            gcs_path = output_path.name

        log.info(f"Uploading shard to GCS: gs://{gcs_client.bucket_name}/{gcs_path}")

        # Upload to GCS
        blob = gcs_client.bucket.blob(gcs_path)
        blob.upload_from_filename(str(output_path))

        # Remove local file
        output_path.unlink()
        log.info(
            f"Shard uploaded to GCS and local file removed: gs://{gcs_client.bucket_name}/{gcs_path}"
        )


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
            ("speaker_weight", pa.float64()),
            ("encoded_audio_shape", pa.list_(pa.int64())),
            ("encoded_audio", pa.list_(pa.int64())),
        ]
    )


def _extract_chunk_data_from_db(
    dia: Dia,
    chunk_data: dict,
    speaker_weights: dict[int, float],
    audio_store: AudioStore,
    data_directory: Path | None,
    gcs_client: GCSClient | None,
) -> dict[str, any]:
    """Extract chunk data from database record for Arrow table."""
    chunk_id = chunk_data["chunk_id"]
    chunk_file_path = chunk_data["chunk_file_path"]

    # Calculate speaker weight for this chunk
    speaker_weight = audio_store.get_chunk_speaker_weight(chunk_id, speaker_weights)

    # Load audio data
    encoded_audio = _load_audio_data(chunk_file_path, data_directory, gcs_client, dia)

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
        "speaker_weight": speaker_weight,
        "encoded_audio_shape": encoded_audio_shape,
        "encoded_audio": encoded_audio_flat,
    }


def _load_audio_data(
    chunk_file_path: str,
    data_directory: Path | None,
    gcs_client: GCSClient | None,
    dia: Dia,
) -> any:
    """Load audio data from local directory or GCS."""
    if data_directory:
        # Load from local directory
        local_path = data_directory / chunk_file_path
        if not local_path.exists():
            raise FileNotFoundError(f"Audio chunk not found at {local_path}")
        return dia.load_audio(local_path)

    elif gcs_client:
        # Load from GCS
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp_file:
            # Download from GCS to temporary file
            blob = gcs_client.bucket.blob(f"chunks/{chunk_file_path}")
            blob.download_to_filename(tmp_file.name)

            # Load audio from temporary file
            encoded_audio = dia.load_audio(Path(tmp_file.name))

            return encoded_audio

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
        help="Local directory containing audio chunk files (if not provided, will fetch from GCS)",
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
    parser.add_argument(
        "--gcs-upload-path",
        "-g",
        type=str,
        help="GCS path to upload shards to (e.g., 'gs://bucket/path/'), removes local files after upload",
    )
    parser.add_argument(
        "--chunks-file",
        "-c",
        type=Path,
        help="JSON file containing chunks data (if provided, skips database query)",
    )
    parser.add_argument(
        "--save-chunks",
        "-s",
        type=Path,
        help="Save chunks to JSON file instead of processing them",
    )
    parser.add_argument(
        "--split-chunks",
        type=Path,
        help="Split chunks file into two equal parts (set1 and set2)",
    )

    args = parser.parse_args()

    # Set up logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Handle split chunks operation
    if args.split_chunks:
        split_chunks_file(args.split_chunks)
        return

    # Handle save chunks operation
    if args.save_chunks:
        audio_store = AudioStore()
        log.info(f"Fetching audio chunks from database (limit: {args.limit})...")
        chunks = audio_store.get_all_audio_chunks(limit=args.limit)
        save_chunks_to_file(chunks, args.save_chunks)
        return

    # Convert to arrow table
    convert_to_arrow_table(
        dia_config_path="config.json",
        output_path=args.output,
        chunk_limit=args.limit,
        data_directory=args.data_directory,
        num_workers=args.workers,
        gcs_upload_path=args.gcs_upload_path,
        chunks_file=args.chunks_file,
    )


if __name__ == "__main__":
    main()
