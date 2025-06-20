from dataclasses import dataclass
from typing import Optional

import torch

from parkiet.dia.config import DiaConfig
from parkiet.dia.state import create_attn_mask


@dataclass
class EncoderTrainingState:
    """Parameters specifically for encoder training."""

    max_seq_len: int
    device: torch.device
    positions: torch.Tensor
    padding_mask: torch.Tensor
    attn_mask: torch.Tensor

    @classmethod
    def new(cls, config: DiaConfig, cond_src: torch.Tensor) -> "EncoderTrainingState":
        """Creates EncoderTrainingState from DiaConfig and conditioning source."""
        device = cond_src.device

        positions = torch.arange(
            config.data.text_length, dtype=torch.float32, device=device
        ).unsqueeze(0)
        padding_mask = (cond_src.squeeze(1) != config.data.text_pad_value).to(device)
        attn_mask = create_attn_mask(
            padding_mask, padding_mask, device, is_causal=False
        )

        return cls(
            max_seq_len=config.data.text_length,
            device=device,
            positions=positions,
            padding_mask=padding_mask,
            attn_mask=attn_mask,
        )


class KVCacheTraining(torch.nn.Module):
    """KV Cache for training - handles single batch size instead of doubled."""

    k: torch.Tensor
    v: torch.Tensor

    def __init__(
        self,
        batch_size: int,
        num_heads: int,
        max_len: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
        k: torch.Tensor | None = None,
        v: torch.Tensor | None = None,
    ):
        k = (
            torch.zeros(
                (batch_size, num_heads, max_len, head_dim),
                dtype=dtype,
                device=device,
            )
            if k is None
            else k
        )
        v = (
            torch.zeros(
                (batch_size, num_heads, max_len, head_dim),
                dtype=dtype,
                device=device,
            )
            if v is None
            else v
        )
        super().__init__()

        self.register_buffer("k", k)
        self.register_buffer("v", v)

    @classmethod
    def from_kv(cls, k: torch.Tensor, v: torch.Tensor) -> "KVCacheTraining":
        return cls(
            batch_size=k.shape[0],
            num_heads=k.shape[1],
            max_len=k.shape[2],
            head_dim=k.shape[3],
            dtype=k.dtype,
            device=k.device,
            k=k,
            v=v,
        )

    def update(
        self, k: torch.Tensor, v: torch.Tensor, current_idx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        k_out, v_out = self.k, self.v
        k_out[:, :, current_idx, :] = k
        v_out[:, :, current_idx, :] = v
        return self.k, self.v

    def prefill(self, k: torch.Tensor, v: torch.Tensor):
        prefill_len = k.shape[2]
        self.k[:, :, :prefill_len, :] = k
        self.v[:, :, :prefill_len, :] = v


@dataclass
class DecoderTrainingState:
    """Parameters specifically for decoder training."""

    device: torch.device
    dtype: torch.dtype
    enc_out: torch.Tensor
    enc_positions: torch.Tensor
    dec_positions: torch.Tensor
    self_attn_cache: list[KVCacheTraining]
    cross_attn_cache: list[KVCacheTraining]
    casual_attn_mask: torch.Tensor
    cross_attn_mask: torch.Tensor

    @classmethod
    def new(
        cls,
        config: DiaConfig,
        enc_state: EncoderTrainingState,
        enc_out: torch.Tensor,
        dec_cross_attn_cache: list[KVCacheTraining],
        compute_dtype: torch.dtype,
        max_generation_length: Optional[int] = None,
    ) -> "DecoderTrainingState":
        """Creates DecoderTrainingState from DiaConfig and encoder state."""
        device = enc_out.device
        max_audio_len = max_generation_length or config.data.audio_length
        batch_size = enc_out.shape[0]

        dec_positions = torch.full(
            (batch_size, 1), fill_value=0, dtype=torch.int32, device=device
        )
        causal_mask = torch.tril(
            torch.ones(max_audio_len, max_audio_len, dtype=torch.bool, device=device)
        )
        dec_mask = torch.ones((batch_size, 1), dtype=torch.bool, device=device)
        cross_attn_mask = create_attn_mask(
            dec_mask, enc_state.padding_mask, device, is_causal=False
        )

        self_attn_cache = [
            KVCacheTraining(
                batch_size,
                config.model.decoder.kv_heads,
                max_audio_len,
                config.model.decoder.gqa_head_dim,
                compute_dtype,
                device,
            )
            for _ in range(config.model.decoder.n_layer)
        ]

        return cls(
            device=device,
            dtype=compute_dtype,
            enc_out=enc_out,
            enc_positions=enc_state.positions,
            dec_positions=dec_positions,
            self_attn_cache=self_attn_cache,
            cross_attn_cache=dec_cross_attn_cache,
            casual_attn_mask=causal_mask,
            cross_attn_mask=cross_attn_mask,
        )

    def prepare_step(self, step_from: int, step_to: int | None = None) -> None:
        if step_to is None:
            step_to = step_from + 1
        self.dec_positions = torch.arange(
            step_from, step_to, dtype=torch.int32, device=self.device
        ).unsqueeze(0)
