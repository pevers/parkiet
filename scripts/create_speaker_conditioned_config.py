from __future__ import annotations

import argparse
import json
from pathlib import Path

from parkiet.dia.config import DiaConfig


def load_speaker_vocab(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def derive_num_speakers(speaker_vocab: dict) -> int:
    model_ids: list[int] = []
    for mapping_key in (
        "voice_id_to_model_speaker_id",
        "source_speaker_id_to_model_speaker_id",
    ):
        for value in speaker_vocab.get(mapping_key, {}).values():
            model_ids.append(int(value))
    if not model_ids:
        raise ValueError("speaker_vocab does not contain any model speaker IDs")
    return max(model_ids) + 1


def build_speaker_conditioned_config(
    base_config: DiaConfig,
    speaker_vocab: dict,
    *,
    speaker_embedding_dim: int = 256,
) -> DiaConfig:
    payload = base_config.model_dump()
    payload["speaker_conditioning_enabled"] = True
    payload["speaker_embedding_dim"] = int(speaker_embedding_dim)
    payload["num_speakers"] = derive_num_speakers(speaker_vocab)
    payload["default_speaker_id"] = int(speaker_vocab["default_speaker_id"])
    payload["speaker_conditioning_mode"] = "encoder_decoder_additive"
    return DiaConfig.model_validate(payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create speaker-conditioned Dia config")
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--speaker-vocab", required=True)
    parser.add_argument("--output-config", required=True)
    parser.add_argument("--speaker-embedding-dim", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_config = DiaConfig.load(args.base_config)
    if base_config is None:
        raise FileNotFoundError(f"Base config not found: {args.base_config}")
    speaker_vocab = load_speaker_vocab(args.speaker_vocab)
    conditioned = build_speaker_conditioned_config(
        base_config,
        speaker_vocab,
        speaker_embedding_dim=args.speaker_embedding_dim,
    )
    conditioned.save(args.output_config)
    print(f"output_config={args.output_config}")
    print(f"speaker_conditioning_enabled={conditioned.speaker_conditioning_enabled}")
    print(f"num_speakers={conditioned.num_speakers}")
    print(f"default_speaker_id={conditioned.default_speaker_id}")
    print(f"speaker_embedding_dim={conditioned.speaker_embedding_dim}")


if __name__ == "__main__":
    main()
