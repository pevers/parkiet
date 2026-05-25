from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from parkiet.dia.config import DiaConfig
from parkiet.dia.model import Dia, load_state_dict_allowing_missing_speaker_modules
from parkiet.jax.dataset import create_dataset


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def summarize_batch(batch: dict[str, Any]) -> dict[str, Any]:
    speaker_ids = batch.get("speaker_id")
    speaker_id_values: list[int] = []
    if speaker_ids is not None:
        speaker_id_values = [int(v) for v in np.array(speaker_ids).reshape(-1).tolist()]
    return {
        "keys": sorted(batch.keys()),
        "batch_size": int(np.array(batch["text"]).shape[0]),
        "speaker_id_values": speaker_id_values,
        "speaker_id_unique": sorted(set(speaker_id_values)),
    }


def load_base_checkpoint_for_conditioned_model(
    config: DiaConfig,
    checkpoint_path: str | Path,
) -> dict[str, Any]:
    dia = Dia(config=config, compute_dtype="float32", load_dac=False)
    state_dict = torch.load(checkpoint_path, map_location=dia.device)
    missing_keys, unexpected_keys = load_state_dict_allowing_missing_speaker_modules(
        dia.model,
        state_dict,
        allow_missing_speaker_modules=config.speaker_conditioning_enabled,
    )
    return {
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
        "initialized_speaker_module_keys": [
            key for key in missing_keys if key.startswith("speaker_")
        ],
    }


def run_smoke(
    *,
    config_path: str | Path,
    checkpoint_path: str | Path,
    parquet_path: str | Path,
    speaker_vocab_path: str | Path,
    max_steps: int = 5,
    batch_size: int = 2,
    learning_rate: float = 1e-5,
) -> dict[str, Any]:
    config = DiaConfig.load(str(config_path))
    if config is None:
        raise FileNotFoundError(f"Config file not found: {config_path}")
    speaker_vocab = load_json(speaker_vocab_path)
    dataset = create_dataset(
        config=config,
        parquet_path=str(parquet_path),
        transcription_clean_prob=0.1,
        text_dropout_prob=0.15,
        speaker_vocab=speaker_vocab,
    )
    batch_iterator = dataset.batch_iterator(
        batch_size=batch_size,
        shuffle=False,
        seed=42,
        use_sample_prob=False,
    )
    batch = next(batch_iterator)
    batch_summary = summarize_batch(batch)
    checkpoint_summary = load_base_checkpoint_for_conditioned_model(
        config,
        checkpoint_path,
    )
    return {
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint_path),
        "parquet_path": str(parquet_path),
        "speaker_vocab_path": str(speaker_vocab_path),
        "speaker_conditioning_enabled": config.speaker_conditioning_enabled,
        "num_speakers": config.num_speakers,
        "default_speaker_id": config.default_speaker_id,
        "max_steps": int(max_steps),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "batch_summary": batch_summary,
        "checkpoint_summary": checkpoint_summary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tiny speaker-conditioned finetune smoke validator")
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--parquet-path", required=True)
    parser.add_argument("--speaker-vocab-path", required=True)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_smoke(
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
        parquet_path=args.parquet_path,
        speaker_vocab_path=args.speaker_vocab_path,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "smoke_report.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"output_report={output_path}")
    print(f"speaker_conditioning_enabled={summary['speaker_conditioning_enabled']}")
    print(f"num_speakers={summary['num_speakers']}")
    print(f"speaker_id_unique={summary['batch_summary']['speaker_id_unique']}")
    print(
        "initialized_speaker_module_keys="
        f"{summary['checkpoint_summary']['initialized_speaker_module_keys']}"
    )


if __name__ == "__main__":
    main()
