from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


def load_speaker_vocab(path: str | Path) -> dict[str, Any]:
    vocab_path = Path(path)
    with vocab_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if "default_speaker_id" not in data:
        raise ValueError("speaker vocab is missing 'default_speaker_id'")
    mapping = data.get("db_speaker_id_to_model_speaker_id")
    if mapping is None:
        mapping = data.get("source_speaker_id_to_model_speaker_id")
    if mapping is None:
        raise ValueError(
            "speaker vocab is missing 'db_speaker_id_to_model_speaker_id' or "
            "'source_speaker_id_to_model_speaker_id'"
        )
    if not isinstance(mapping, dict):
        raise ValueError("speaker id mapping must be an object")
    return {
        "default_speaker_id": int(data["default_speaker_id"]),
        "db_speaker_id_to_model_speaker_id": {
            str(key): int(value) for key, value in mapping.items()
        },
    }


def save_speaker_vocab(path: str | Path, mapping: dict[str, Any]) -> None:
    vocab_path = Path(path)
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    normalized = {
        "default_speaker_id": int(mapping.get("default_speaker_id", 0)),
        "db_speaker_id_to_model_speaker_id": {
            str(key): int(value)
            for key, value in mapping.get("db_speaker_id_to_model_speaker_id", {}).items()
        },
    }
    vocab_path.write_text(json.dumps(normalized, indent=2), encoding="utf-8")


def build_speaker_vocab_from_chunk_owners(
    chunk_owners: list[int | None], min_count: int = 1
) -> dict[str, Any]:
    if min_count < 1:
        raise ValueError("min_count must be >= 1")
    counts = Counter(owner for owner in chunk_owners if owner is not None and owner >= 0)
    sorted_owners = sorted(owner for owner, count in counts.items() if count >= min_count)
    mapping = {
        str(owner): index
        for index, owner in enumerate(sorted_owners, start=1)
    }
    return {
        "default_speaker_id": 0,
        "db_speaker_id_to_model_speaker_id": mapping,
    }


def map_chunk_owner_to_speaker_id(
    chunk_owner: int | None,
    vocab: dict[str, Any],
    default_speaker_id: int = 0,
) -> int:
    if chunk_owner is None or chunk_owner < 0:
        return int(default_speaker_id)
    mapping = vocab.get("db_speaker_id_to_model_speaker_id", {})
    return int(mapping.get(str(chunk_owner), default_speaker_id))
