"""
Simple script to view parquet dataset contents showing transcriptions and audio paths.
"""

import argparse
import logging
from pathlib import Path
import pyarrow.parquet as pq
import pandas as pd
from glob import glob

log = logging.getLogger(__name__)


def view_parquet_dataset(
    parquet_path: Path,
    limit: int = 10,
    show_audio: bool = False,
    summary_only: bool = False,
) -> None:
    """View contents of parquet dataset files.

    Args:
        parquet_path: Path to parquet file or directory containing parquet files
        limit: Maximum number of rows to display
        show_audio: Whether to show encoded audio data (very verbose)
        summary_only: Only show summary statistics, no individual rows
    """
    # Find all parquet files
    parquet_files = []

    if parquet_path.is_file():
        parquet_files = [parquet_path]
    elif parquet_path.is_dir():
        parquet_files = list(parquet_path.glob("*.parquet"))
        parquet_files.sort()
    else:
        # Try glob pattern
        parquet_files = [Path(f) for f in glob(str(parquet_path))]
        parquet_files.sort()

    if not parquet_files:
        log.error(f"No parquet files found at: {parquet_path}")
        return

    log.info(f"Found {len(parquet_files)} parquet file(s)")

    # Read and display each file
    total_rows_shown = 0
    for file_path in parquet_files:
        if total_rows_shown >= limit:
            break

        log.info(f"\n{'=' * 60}")
        log.info(f"File: {file_path}")
        log.info(f"{'=' * 60}")

        # Read parquet file
        table = pq.read_table(file_path)
        df = table.to_pandas()

        log.info(f"Shape: {df.shape}")
        log.info(f"Columns: {list(df.columns)}")

        # Show sample data only if not summary_only
        if not summary_only:
            rows_to_show = min(limit - total_rows_shown, len(df))
            sample_df = df.head(rows_to_show)

            log.info(f"\nSample data (showing {rows_to_show} rows):")
            log.info("-" * 80)

            for idx, row in sample_df.iterrows():
                log.info(f"\nRow {idx + 1}:")
                log.info(f"  Chunk ID: {row['chunk_id']}")
                log.info(f"  Source File: {row['source_file']}")
                log.info(f"  Audio Path: {row['file_path']}")
                log.info(f"  Duration: {row['duration_ms']:.1f}ms")
                log.info(f"  CB Weight: {row['cb_weight']:.6f}")
                log.info(f"  Sample Prob: {row['sample_prob']:.6f}")

                # Show transcriptions
                if pd.notna(row["transcription"]) and row["transcription"].strip():
                    log.info(f"  Transcription: {row['transcription']}")
                if (
                    pd.notna(row["transcription_clean"])
                    and row["transcription_clean"].strip()
                ):
                    log.info(f"  Clean Transcription: {row['transcription_clean']}")

                # Show audio info
                log.info(f"  Audio Shape: {row['encoded_audio_shape']}")
                if show_audio:
                    audio_data = row["encoded_audio"]
                    log.info(f"  Audio Data (first 10 values): {audio_data[:10]}")

            total_rows_shown += rows_to_show

        # Show summary stats
        log.info(f"\nSummary for {file_path.name}:")
        log.info(f"  Total chunks: {len(df)}")
        log.info(
            f"  Duration range: {df['duration_ms'].min():.1f} - {df['duration_ms'].max():.1f}ms"
        )
        log.info(
            f"  CB weight range: {df['cb_weight'].min():.3f} - {df['cb_weight'].max():.3f}"
        )
        log.info(
            f"  Sample prob range: {df['sample_prob'].min():.6f} - {df['sample_prob'].max():.6f}"
        )

        # CB weight distribution
        cb_weights = df["cb_weight"].round(3)
        cb_counts = cb_weights.value_counts().sort_index()
        log.info(f"  CB weight distribution:")
        for weight, count in cb_counts.items():
            percentage = (count / len(df)) * 100
            log.info(f"    {weight:.3f}: {count} chunks ({percentage:.1f}%)")

        log.info(
            f"  Non-empty transcriptions: {df['transcription'].str.strip().astype(bool).sum()}"
        )
        log.info(f"  Unique source files: {df['source_file'].nunique()}")


def main():
    """CLI entry point for the parquet viewer."""
    parser = argparse.ArgumentParser(
        description="View parquet dataset contents showing transcriptions and audio paths"
    )
    parser.add_argument(
        "parquet_path",
        type=Path,
        help="Path to parquet file, directory, or glob pattern (e.g., 'dataset_*.parquet')",
    )
    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=10,
        help="Maximum number of rows to display (default: 10)",
    )
    parser.add_argument(
        "--show-audio",
        "-a",
        action="store_true",
        help="Show encoded audio data (very verbose)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )
    parser.add_argument(
        "--summary-only",
        "-s",
        action="store_true",
        help="Only show summary statistics, no individual rows",
    )
    args = parser.parse_args()

    # Set up logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",  # Simple format for better readability
    )

    # View parquet dataset
    view_parquet_dataset(
        parquet_path=args.parquet_path,
        limit=args.limit,
        show_audio=args.show_audio,
        summary_only=args.summary_only,
    )


if __name__ == "__main__":
    main()
