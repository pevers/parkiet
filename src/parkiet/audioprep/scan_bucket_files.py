"""
Script to scan and list all MP3 files in the GCS bucket.
Scans the entire bucket and saves the results to a file for later processing.
"""

import argparse
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from parkiet.storage.gcs_client import get_gcs_client

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)


def scan_bucket_mp3s(output_file: str = "bucket_mp3_files.txt") -> int:
    """
    Scan all MP3 files in the GCS bucket and save to file.

    Args:
        output_file: Output file path

    Returns:
        Number of MP3 files found
    """
    try:
        gcs_client = get_gcs_client()
        log.info(f"Scanning bucket: {gcs_client.bucket_name}")

        # List all blobs in the bucket
        blobs = gcs_client.bucket.list_blobs()

        mp3_files = []
        total_files = 0

        for blob in blobs:
            total_files += 1
            if blob.name.lower().endswith(".mp3"):
                mp3_files.append(blob.name)
                if len(mp3_files) % 100 == 0:
                    log.info(f"Found {len(mp3_files)} MP3 files so far...")

        log.info(f"Scanned {total_files} total files, found {len(mp3_files)} MP3 files")

        # Sort files for consistent output
        mp3_files.sort()

        # Write to file with gs:// prefix for compatibility with queue_files.py
        output_path = Path(output_file)
        with open(output_path, "w", encoding="utf-8") as f:
            for file_path in mp3_files:
                f.write(f"gs://{gcs_client.bucket_name}/{file_path}\n")

        log.info(f"Saved {len(mp3_files)} MP3 file paths to {output_path}")
        return len(mp3_files)

    except Exception as e:
        log.error(f"Error scanning bucket MP3s: {e}")
        return 0


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description="Scan all MP3 files in the GCS bucket")
    parser.add_argument(
        "--output-file",
        "-o",
        default="bucket_mp3_files.txt",
        help="Output file path (default: bucket_mp3_files.txt)",
    )

    args = parser.parse_args()

    count = scan_bucket_mp3s(args.output_file)
    if count > 0:
        log.info(f"Successfully scanned {count} MP3 files")
    else:
        log.error("Failed to scan MP3 files")


if __name__ == "__main__":
    main()
