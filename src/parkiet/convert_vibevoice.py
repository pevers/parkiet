"""Converts the Parquet files to JSONL suitable for VibeVoice."""

import argparse
import json
import logging
from pathlib import Path
from google.cloud.exceptions import NotFound
import pandas as pd
from parkiet.storage.gcs_client import GCSClient

log = logging.getLogger(__name__)


def discover_parquet_files(path: Path) -> list[Path]:
    """Discover parquet files from a file or directory.

    Args:
        path: Path to a parquet file or directory containing parquet files

    Returns:
        List of parquet file paths
    """
    if path.is_file():
        if path.suffix == ".parquet":
            return [path]
        else:
            raise ValueError(f"File {path} is not a parquet file")
    elif path.is_dir():
        parquet_files = list(path.glob("*.parquet"))
        parquet_files.sort()
        return parquet_files
    else:
        raise FileNotFoundError(f"Path {path} does not exist")


def convert_to_vibevoice(
    parquet_path: Path,
    output_dir: Path,
    limit: int | None = None,
) -> None:
    """Convert parquet files to JSONL format for VibeVoice.

    Args:
        parquet_path: Path to parquet file or directory containing parquet files
        output_dir: Directory where to save JSONL files and audio chunks
        limit: Optional limit on number of samples to process
    """
    # Discover parquet files
    parquet_files = discover_parquet_files(parquet_path)
    log.info(
        f"Found {len(parquet_files)} parquet file(s): {[f.name for f in parquet_files]}"
    )

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(exist_ok=True)

    # Initialize GCS client
    gcs_client = GCSClient()

    # Process each parquet file
    total_processed = 0
    jsonl_path = output_dir / "data.jsonl"

    with open(jsonl_path, "w") as jsonl_file:
        for parquet_file in parquet_files:
            log.info(f"Processing {parquet_file}...")
            df = pd.read_parquet(parquet_file)

            for idx, row in df.iterrows():
                if limit is not None and total_processed >= limit:
                    log.info(f"Reached limit of {limit} samples")
                    break

                # Extract transcript and chunk path
                transcript = row["transcription"]
                chunk_path = row["file_path"]
                chunk_id = row["chunk_id"]

                # Download audio chunk from GCS
                gcs_path = f"chunks/{chunk_path}"
                local_audio_path = audio_dir / f"{chunk_id}.wav"

                try:
                    # Download from GCS
                    blob = gcs_client.bucket.blob(gcs_path)
                    blob.download_to_filename(str(local_audio_path))

                    # Write JSONL entry
                    entry = {
                        "audio_path": str(local_audio_path.relative_to(output_dir)),
                        "transcript": transcript,
                        "chunk_id": chunk_id,
                    }
                    jsonl_file.write(json.dumps(entry) + "\n")

                    total_processed += 1
                    if total_processed % 10 == 0:
                        log.info(f"Processed {total_processed} samples...")

                except NotFound:
                    log.warning(f"Could not find {gcs_path} in GCS, skipping")
                    continue
                except Exception as e:
                    log.error(f"Error processing {chunk_id}: {e}")
                    continue

            if limit is not None and total_processed >= limit:
                break

    log.info(f"Conversion complete! Processed {total_processed} samples")
    log.info(f"JSONL file: {jsonl_path}")
    log.info(f"Audio files: {audio_dir}")


def main():
    """CLI entry point for the VibeVoice converter."""
    parser = argparse.ArgumentParser(
        description="Convert parquet files to JSONL format for VibeVoice"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Path to parquet file or directory containing parquet files",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("vibevoice_data"),
        help="Output directory for JSONL and audio files (default: vibevoice_data)",
    )
    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        help="Limit number of samples to process (for testing)",
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

    # Convert to VibeVoice format
    convert_to_vibevoice(
        parquet_path=args.input,
        output_dir=args.output,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
