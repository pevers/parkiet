import json
import logging
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from parkiet.audioprep.schemas import ProcessedAudioFile, ProcessedAudioChunk
from parkiet.dia.model import Dia

log = logging.getLogger(__name__)


def convert_to_arrow_table(dia: Dia, target_folder: Path) -> None:
    """Convert processed audio files to Apache Arrow format.

    Args:
        target_folder: Folder containing processed audio data
    """
    json_files = list(target_folder.glob("*/processed_file.json"))

    if not json_files:
        log.warning("No processed files found to convert")
        return

    log.info(f"Found {len(json_files)} processed files to convert")

    # Collect all chunk data
    # TODO: We should process in parallel and we should prevent huge memory usage
    all_chunk_data = []
    total_chunks = 0
    for json_file in json_files:
        log.info(f"Processing {json_file}")
        try:
            with open(json_file, "r") as f:
                file_data = json.load(f)

            processed_file = ProcessedAudioFile(**file_data)
            if not processed_file.success:
                log.warning(f"Skipping failed file: {json_file}")
                continue

            for chunk in processed_file.chunks:
                chunk_data = _extract_chunk_data(dia, processed_file, chunk)
                all_chunk_data.append(chunk_data)
                total_chunks += 1

        except Exception as e:
            log.error(f"Error processing {json_file}: {e}")
            continue

    if not all_chunk_data:
        log.warning("No valid chunk data found")
        return

    log.info(f"Creating Arrow table with {total_chunks} chunks...")
    column_data = {}
    schema = _define_arrow_schema()

    for field in schema:
        column_data[field.name] = []

    # Populate column data from row data
    for row in all_chunk_data:
        for column_name in column_data.keys():
            column_data[column_name].append(row[column_name])

    table = pa.table(column_data, schema=schema)

    arrow_path = target_folder / "chunks_dataset.parquet"
    pq.write_table(table, arrow_path, compression="snappy")

    log.info(f"Arrow table saved to: {arrow_path}")
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
            ("file_path", pa.string()),
            ("encoded_audio_shape", pa.list_(pa.int64())),
            ("encoded_audio", pa.list_(pa.int64())),
        ]
    )


def _extract_chunk_data(
    dia: Dia,
    processed_file: ProcessedAudioFile,
    chunk: ProcessedAudioChunk,
) -> dict[str, any]:
    """Extract chunk data for Arrow table."""
    output_dir = Path(processed_file.output_directory)
    chunk_file_path = output_dir / chunk.audio_chunk.file_path
    encoded_audio = dia.load_audio(chunk_file_path)
    encoded_audio_shape = list(encoded_audio.shape)
    encoded_audio_flat = encoded_audio.flatten().tolist()

    return {
        "source_file": processed_file.source_file,
        "chunk_id": Path(chunk.audio_chunk.file_path).stem,
        "start_ms": chunk.audio_chunk.start,
        "end_ms": chunk.audio_chunk.end,
        "duration_ms": chunk.audio_chunk.end - chunk.audio_chunk.start,
        "transcription": chunk.transcription,
        "file_path": chunk.audio_chunk.file_path,
        "encoded_audio_shape": encoded_audio_shape,
        "encoded_audio": encoded_audio_flat,
    }
