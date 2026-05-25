from __future__ import annotations

import json
from pathlib import Path

import torch

from parkiet.dia.config import DecoderConfig, DiaConfig, EncoderConfig
from parkiet.dia.layers import DiaModel
from parkiet.dia.model import load_state_dict_allowing_missing_speaker_modules
from scripts.create_speaker_conditioned_config import (
    build_speaker_conditioned_config,
    derive_num_speakers,
)


def _base_config() -> DiaConfig:
    return DiaConfig(
        encoder_config=EncoderConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=8,
        ),
        decoder_config=DecoderConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            cross_hidden_size=16,
            cross_num_attention_heads=2,
            cross_num_key_value_heads=2,
            cross_head_dim=8,
            max_position_embeddings=8,
            num_channels=9,
            vocab_size=32,
        ),
        delay_pattern=(0, 8, 9, 10, 11, 12, 13, 14, 15),
        bos_token_id=30,
        eos_token_id=31,
        pad_token_id=29,
    )


def _speaker_vocab() -> dict:
    return {
        "speaker_vocab_version": "v1",
        "default_speaker_id": 0,
        "voice_id_to_model_speaker_id": {
            "nl_default_01": 0,
            "nl_speaker_0001": 1,
        },
        "source_speaker_id_to_model_speaker_id": {
            "1": 1,
        },
    }


def test_conditioned_config_gets_expected_speaker_fields():
    conditioned = build_speaker_conditioned_config(
        _base_config(),
        _speaker_vocab(),
        speaker_embedding_dim=256,
    )
    assert conditioned.speaker_conditioning_enabled is True
    assert conditioned.num_speakers == 2
    assert conditioned.default_speaker_id == 0
    assert conditioned.speaker_embedding_dim == 256
    assert conditioned.speaker_conditioning_mode == "encoder_decoder_additive"


def test_derive_num_speakers_uses_max_model_speaker_id_plus_one():
    assert derive_num_speakers(_speaker_vocab()) == 2


def test_relaxed_loader_allows_only_missing_speaker_module_keys():
    base_model = DiaModel(_base_config(), torch.float32)
    conditioned_config = build_speaker_conditioned_config(_base_config(), _speaker_vocab())
    conditioned_model = DiaModel(conditioned_config, torch.float32)
    state_dict = base_model.state_dict()

    missing_keys, unexpected_keys = load_state_dict_allowing_missing_speaker_modules(
        conditioned_model,
        state_dict,
        allow_missing_speaker_modules=True,
    )
    assert unexpected_keys == []
    assert sorted(missing_keys) == sorted(
        [
            "speaker_embedding.weight",
            "speaker_to_encoder.weight",
            "speaker_to_encoder.bias",
            "speaker_to_decoder.weight",
            "speaker_to_decoder.bias",
        ]
    )


def test_relaxed_loader_rejects_non_speaker_missing_keys():
    base_model = DiaModel(_base_config(), torch.float32)
    conditioned_config = build_speaker_conditioned_config(_base_config(), _speaker_vocab())
    conditioned_model = DiaModel(conditioned_config, torch.float32)
    state_dict = base_model.state_dict()
    state_dict.pop("encoder.norm.weight")

    try:
        load_state_dict_allowing_missing_speaker_modules(
            conditioned_model,
            state_dict,
            allow_missing_speaker_modules=True,
        )
    except RuntimeError as exc:
        assert "non-speaker keys" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError for missing non-speaker keys")


def test_speaker_vocab_maps_source_speaker_1_to_model_speaker_1(tmp_path: Path):
    vocab_path = tmp_path / "speaker_vocab.json"
    vocab_path.write_text(json.dumps(_speaker_vocab()), encoding="utf-8")
    loaded = json.loads(vocab_path.read_text(encoding="utf-8"))
    assert loaded["source_speaker_id_to_model_speaker_id"]["1"] == 1
