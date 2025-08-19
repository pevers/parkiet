import pandas as pd
from pathlib import Path
import sys


def main():
    if len(sys.argv) != 2:
        print("Usage: python show_clean_transcripts.py <parquet_file>")
        sys.exit(1)

    parquet_path = Path(sys.argv[1])
    if not parquet_path.exists():
        print(f"File not found: {parquet_path}")
        sys.exit(1)

    # Load the parquet file
    df = pd.read_parquet(parquet_path)

    # Filter for rows with clean transcripts
    clean_df = df[
        (df["transcription_clean"].notna()) & (df["transcription_clean"].str.len() > 0)
    ]

    print(
        f"Found {len(clean_df)} samples with clean transcripts out of {len(df)} total samples"
    )
    print("\nClean transcripts:")
    print("-" * 80)

    for idx, row in clean_df.iterrows():
        print(f"Sample {idx}:")
        print(f"Original: {row['transcription']}")
        print(f"Clean:    {row['transcription_clean']}")
        print("-" * 80)


if __name__ == "__main__":
    main()
