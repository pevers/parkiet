import argparse
import logging
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from parkiet.database.audio_store import AudioStore
from parkiet.dia.config import DiaConfig
from parkiet.storage.gcs_client import GCSClient
from parkiet.dia.model import ComputeDtype, Dia
import tempfile
import os

log = logging.getLogger(__name__)


def convert_to_arrow_table(
    dia: Dia,
    output_path: Path,
    chunk_limit: int | None = None,
    data_directory: Path | None = None,
) -> None:
    """Convert processed audio chunks from database to Apache Arrow format.

    Args:
        dia: Dia model for audio processing
        output_path: Path where to save the parquet file
        chunk_limit: Optional limit on number of chunks to process
        data_directory: Local directory containing audio chunks, if None uses GCS
    """
    audio_store = AudioStore()

    # Get all audio chunks from database
    log.info(f"Fetching audio chunks from database (limit: {chunk_limit})...")
    chunks = audio_store.get_all_audio_chunks(limit=chunk_limit)

    if not chunks:
        log.warning("No audio chunks found in database")
        return

    log.info(f"Found {len(chunks)} audio chunks to process")

    # Pre-calculate speaker weights
    log.info("Calculating speaker weights...")
    speaker_weights = audio_store.calculate_speaker_weights()
    log.info(f"Calculated weights for {len(speaker_weights)} speakers")

    # Initialize GCS client if no local data directory
    gcs_client = None
    if data_directory is None:
        log.info("No data directory provided, will fetch audio from GCS")
        gcs_client = GCSClient()

    # Collect all chunk data
    all_chunk_data = []
    total_processed = 0

    for chunk in chunks:
        log.debug(f"Processing chunk {chunk['chunk_id']}")
        chunk_data = _extract_chunk_data_from_db(
            dia, chunk, speaker_weights, audio_store, data_directory, gcs_client
        )
        all_chunk_data.append(chunk_data)
        total_processed += 1

        if total_processed % 100 == 0:
            log.info(f"Processed {total_processed}/{len(chunks)} chunks")

    if not all_chunk_data:
        log.warning("No valid chunk data found")
        return

    log.info(f"Creating Arrow table with {total_processed} chunks...")
    column_data = {}
    schema = _define_arrow_schema()

    for field in schema:
        column_data[field.name] = []

    # Populate column data from row data
    for row in all_chunk_data:
        for column_name in column_data.keys():
            column_data[column_name].append(row[column_name])

    table = pa.table(column_data, schema=schema)

    pq.write_table(table, output_path, compression="snappy")

    log.info(f"Arrow table saved to: {output_path}")
    log.info(f"Table shape: {table.shape}")


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
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            try:
                # Download from GCS to temporary file
                blob = gcs_client.bucket.blob(f"chunks/{chunk_file_path}")
                blob.download_to_filename(tmp_file.name)

                # Load audio from temporary file
                encoded_audio = dia.load_audio(Path(tmp_file.name))

                return encoded_audio
            finally:
                # Clean up temporary file
                if os.path.exists(tmp_file.name):
                    os.unlink(tmp_file.name)

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
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Set up logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Initialize Dia model
    log.info("Initializing Dia model...")
    dia_config = DiaConfig.load("config.json")
    dia = Dia(config=dia_config, compute_dtype=ComputeDtype.BFLOAT16, load_dac=True)

    # Hack to stay compatible with the old model
    dia._load_dac_model()

    # Convert to arrow table
    convert_to_arrow_table(
        dia=dia,
        output_path=args.output,
        chunk_limit=args.limit,
        data_directory=args.data_directory,
    )


if __name__ == "__main__":
    main()
