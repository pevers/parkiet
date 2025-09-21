import jax.numpy as jnp
from flax.traverse_util import flatten_dict, unflatten_dict
import flax.nnx as nnx
from flax.nnx.statelib import replace_by_pure_dict
import logging
import time
import numpy as np
from enum import Enum
import orbax.checkpoint as ocp
import jax

from jax import random

import torch
from .audio import (
    build_delay_indices,
    apply_audio_delay,
    build_revert_indices,
    revert_audio_delay,
)

from parkiet.dia.config import DiaConfig
from parkiet.jax.layers import DiaModel
from parkiet.jax.state import (
    DecoderInferenceState,
    DecoderOutput,
    EncoderInferenceState,
    KVCache,
)

log = logging.getLogger(__name__)

DEFAULT_SAMPLE_RATE = 44100
SAMPLE_RATE_RATIO = 512


def torch_to_flax_weight(torch_tensor: torch.Tensor):
    # Convert torch tensor to numpy, then to jax array
    # Handle BFloat16 by converting to float32 first
    if torch_tensor.dtype == torch.bfloat16:
        torch_tensor = torch_tensor.float()
    return jnp.array(torch_tensor.cpu().numpy())


def convert_torch_to_nnx(torch_model, nnx_model, dia_config: DiaConfig):
    """
    Map PyTorch model weights to nnx model weights.
    """
    torch_state = torch_model.state_dict()
    graphdef, state = nnx.split(nnx_model)
    state_dict = nnx.to_pure_dict(state)
    state_dict = flatten_dict(state_dict)
    new_flat = {}

    for k, v in state_dict.items():
        torch_key = ".".join([str(s) for s in k])
        torch_key = torch_key.replace("embedding.embedding", "embedding.weight")
        torch_key = torch_key.replace("pre_sa_norm.scale", "pre_sa_norm.weight")
        torch_key = torch_key.replace("post_sa_norm.scale", "post_sa_norm.weight")
        torch_key = torch_key.replace("pre_ca_norm.scale", "pre_ca_norm.weight")
        torch_key = torch_key.replace("pre_mlp_norm.scale", "pre_mlp_norm.weight")
        torch_key = torch_key.replace("q_proj.kernel", "q_proj.weight")
        torch_key = torch_key.replace("k_proj.kernel", "k_proj.weight")
        torch_key = torch_key.replace("v_proj.kernel", "v_proj.weight")
        torch_key = torch_key.replace("o_proj.kernel", "o_proj.weight")
        torch_key = torch_key.replace("c_proj.kernel", "c_proj.weight")
        torch_key = torch_key.replace("wi_fused.kernel", "wi_fused.weight")
        torch_key = torch_key.replace("wo.kernel", "wo.weight")
        torch_key = torch_key.replace("norm.scale", "norm.weight")
        torch_key = torch_key.replace("logits_dense.kernel", "logits_dense.weight")

        # Handle decoder embeddings
        for i in range(0, dia_config.decoder_config.num_channels):
            torch_key = torch_key.replace(
                f"embeddings.{i}.embedding", f"embeddings.{i}.weight"
            )

        # Regular parameter handling
        found = False
        for tk in torch_state:
            if tk == torch_key:
                arr = torch_state[tk]
                new_flat[k] = torch_to_flax_weight(arr)
                found = True
                break
        if not found:
            log.warning(f"Weight {k} not found in PyTorch state (JAX shape: {v.shape})")

    new_params = unflatten_dict(new_flat)
    replace_by_pure_dict(state, new_params)
    return nnx.merge(graphdef, state)


def convert_jax_to_torch(nnx_model, torch_model, dia_config: DiaConfig):
    """
    Map nnx model weights to PyTorch model weights.
    """
    _, state = nnx.split(nnx_model)
    state_dict = nnx.to_pure_dict(state)
    state_dict = flatten_dict(state_dict)

    torch_state = torch_model.state_dict()
    new_torch_state = {}

    for k, v in state_dict.items():
        torch_key = ".".join([str(s) for s in k])
        torch_key = torch_key.replace("embedding.embedding", "embedding.weight")
        torch_key = torch_key.replace("pre_sa_norm.scale", "pre_sa_norm.weight")
        torch_key = torch_key.replace("post_sa_norm.scale", "post_sa_norm.weight")
        torch_key = torch_key.replace("pre_ca_norm.scale", "pre_ca_norm.weight")
        torch_key = torch_key.replace("pre_mlp_norm.scale", "pre_mlp_norm.weight")
        torch_key = torch_key.replace("q_proj.kernel", "q_proj.weight")
        torch_key = torch_key.replace("k_proj.kernel", "k_proj.weight")
        torch_key = torch_key.replace("v_proj.kernel", "v_proj.weight")
        torch_key = torch_key.replace("o_proj.kernel", "o_proj.weight")
        torch_key = torch_key.replace("c_proj.kernel", "c_proj.weight")
        torch_key = torch_key.replace("wi_fused.kernel", "wi_fused.weight")
        torch_key = torch_key.replace("wo.kernel", "wo.weight")
        torch_key = torch_key.replace("norm.scale", "norm.weight")
        torch_key = torch_key.replace("logits_dense.kernel", "logits_dense.weight")

        # Handle decoder embeddings
        for i in range(0, dia_config.decoder_config.num_channels):
            torch_key = torch_key.replace(
                f"embeddings.{i}.embedding", f"embeddings.{i}.weight"
            )

        # Skip rotary embedding timescale parameters
        if "rotary_emb.timescale" in torch_key:
            continue

        # Convert JAX array to torch tensor
        if torch_key in torch_state:
            jax_array = v
            # Convert to numpy first, then to torch tensor
            numpy_array = np.array(jax_array)
            torch_tensor = torch.from_numpy(numpy_array)

            # Match the dtype of the original torch parameter
            original_dtype = torch_state[torch_key].dtype
            if torch_tensor.dtype != original_dtype:
                print(
                    f"Converting {torch_key} from {torch_tensor.dtype} to {original_dtype}"
                )
                torch_tensor = torch_tensor.to(original_dtype)

            new_torch_state[torch_key] = torch_tensor
        else:
            log.warning(f"Weight {torch_key} not found in PyTorch model")

    # Load the new state dict into the torch model
    torch_model.load_state_dict(new_torch_state, strict=True)
    return torch_model


def _sample_next_token(
    logits_BCxV: jnp.ndarray,
    temperature: float,
    top_p: float,
    top_k: int | None,
    audio_eos_value: int,
    step_key: jax.Array,
) -> jnp.ndarray:
    if temperature == 0.0:
        argmax_tokens = jnp.argmax(logits_BCxV, axis=-1)
        return argmax_tokens

    logits_BCxV = logits_BCxV / temperature

    if audio_eos_value is not None and audio_eos_value >= 0:
        top_logit_indices_BC = jnp.argmax(logits_BCxV, axis=-1)
        eos_not_highest_mask_BC = top_logit_indices_BC != audio_eos_value

        # Create mask for EOS token where it's not the highest
        eos_not_highest_expanded = eos_not_highest_mask_BC[:, None]
        eos_token_mask = jnp.arange(logits_BCxV.shape[1]) == audio_eos_value
        mask_eos_unless_highest_BCxV = eos_not_highest_expanded & eos_token_mask
        logits_BCxV = jnp.where(mask_eos_unless_highest_BCxV, -jnp.inf, logits_BCxV)

        # Create mask for tokens before EOS when EOS is highest
        eos_highest_mask_BC = top_logit_indices_BC == audio_eos_value
        eos_highest_expanded = eos_highest_mask_BC[:, None]
        before_eos_mask = jnp.arange(logits_BCxV.shape[1]) < audio_eos_value
        mask_eos_highest_BCxV = eos_highest_expanded & before_eos_mask
        logits_BCxV = jnp.where(mask_eos_highest_BCxV, -jnp.inf, logits_BCxV)

    if top_k is not None:
        _, top_k_indices_BCxV = jax.lax.top_k(logits_BCxV, k=top_k)
        # Create mask using broadcasting instead of boolean indexing
        vocab_idx = jnp.arange(logits_BCxV.shape[1])[None, :]

        # Check if each vocab index is in the top-k for each batch
        top_k_mask = jnp.any(
            vocab_idx[:, :, None] == top_k_indices_BCxV[:, None, :], axis=-1
        )
        logits_BCxV = jnp.where(~top_k_mask, -jnp.inf, logits_BCxV)

    if top_p < 1.0:
        probs_BCxV = jax.nn.softmax(logits_BCxV, axis=-1)
        sorted_probs_BCxV = jnp.sort(probs_BCxV, axis=-1)[:, ::-1]
        sorted_indices_BCxV = jnp.argsort(probs_BCxV, axis=-1)[:, ::-1]
        cumulative_probs_BCxV = jnp.cumsum(sorted_probs_BCxV, axis=-1)

        sorted_indices_to_remove_BCxV = cumulative_probs_BCxV > top_p
        sorted_indices_to_remove_BCxV = jnp.roll(
            sorted_indices_to_remove_BCxV, shift=1, axis=-1
        )
        sorted_indices_to_remove_BCxV = sorted_indices_to_remove_BCxV.at[:, 0].set(
            False
        )

        # Create inverse mapping from sorted indices back to original positions using vectorized operations
        vocab_size = logits_BCxV.shape[1]
        vocab_idx = jnp.arange(vocab_size)[None, None, :]
        sorted_idx_expanded = sorted_indices_BCxV[:, :, None]

        # Create mask where each position matches its sorted index
        position_matches = vocab_idx == sorted_idx_expanded
        # Broadcast the removal mask to match the position_matches shape
        should_remove_expanded = sorted_indices_to_remove_BCxV[:, :, None]
        # Combine: if position matches a sorted index that should be removed, mark for removal
        indices_to_remove_BCxV = jnp.any(
            position_matches & should_remove_expanded, axis=1
        )

        logits_BCxV = jnp.where(indices_to_remove_BCxV, -jnp.inf, logits_BCxV)

    final_probs_BCxV = jax.nn.softmax(logits_BCxV, axis=-1)

    sampled_indices_BC = random.categorical(
        step_key, jnp.log(final_probs_BCxV), axis=-1
    )

    return sampled_indices_BC


class ComputeDtype(str, Enum):
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"

    def to_dtype(self) -> jnp.dtype:
        if self == ComputeDtype.FLOAT32:
            return jnp.float32
        elif self == ComputeDtype.FLOAT16:
            return jnp.float16
        elif self == ComputeDtype.BFLOAT16:
            return jnp.bfloat16
        else:
            raise ValueError(f"Unsupported compute dtype: {self}")


class Dia:
    def __init__(
        self,
        config: DiaConfig,
        compute_dtype: str | ComputeDtype = ComputeDtype.FLOAT32,
        param_dtype: str | ComputeDtype = ComputeDtype.FLOAT32,
        load_dac: bool = True,
    ):
        """Initializes the Dia model.

        Args:
            config: The configuration object for the model.
            compute_dtype: The computation dtype to use.
            load_dac: Whether to load the DAC model.

        Raises:
            RuntimeError: If there is an error loading the DAC model.
        """
        self.config = config
        if isinstance(compute_dtype, str):
            compute_dtype = ComputeDtype(compute_dtype)
        if isinstance(param_dtype, str):
            param_dtype = ComputeDtype(param_dtype)
        self.compute_dtype = compute_dtype.to_dtype()
        self.param_dtype = param_dtype.to_dtype()
        self.model: DiaModel = DiaModel(config, self.compute_dtype, self.param_dtype)
        self.dac_model = None
        self.load_dac = load_dac

        if not self.load_dac:
            print("Warning: DAC model will not be loaded. This is not recommended.")

    @classmethod
    def from_local(
        cls,
        config_path: str,
        checkpoint_path: str,
        compute_dtype: str | ComputeDtype = ComputeDtype.FLOAT32,
        param_dtype: str | ComputeDtype = ComputeDtype.FLOAT32,
        load_dac: bool = True,
    ) -> "Dia":
        """Loads the Dia model from local configuration and checkpoint files using orbax.

        Args:
            config_path: Path to the configuration JSON file.
            checkpoint_path: Path to the model checkpoint directory (for orbax).
            compute_dtype: The computation dtype to use.
            load_dac: Whether to load the DAC model.

        Returns:
            An instance of the Dia model loaded with weights.

        Raises:
            FileNotFoundError: If the config or checkpoint file is not found.
            RuntimeError: If there is an error loading the checkpoint.
        """
        config = DiaConfig.load(config_path)
        if config is None:
            raise FileNotFoundError(f"Config file not found at {config_path}")

        dia = cls(config, compute_dtype, param_dtype, load_dac)

        try:
            # Use orbax to load checkpoint
            with ocp.StandardCheckpointer() as checkpointer:
                restored_params = checkpointer.restore(checkpoint_path)

            # Update model parameters
            graphdef, state = nnx.split(dia.model)
            nnx.replace_by_pure_dict(state, restored_params)
            dia.model = nnx.merge(graphdef, state)

        except FileNotFoundError:
            raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")
        except Exception as e:
            raise RuntimeError(
                f"Error loading checkpoint from {checkpoint_path}"
            ) from e

        if load_dac:
            dia._load_dac_model()
        return dia

    def _load_dac_model(self):
        """Loads the Descript Audio Codec (DAC) model.

        Downloads the DAC model if necessary and loads it.
        Sets the DAC model to evaluation mode.

        NOTE: This still uses Torch as there is no proper port of the DAC model to JAX.
        In training we use pre-processed data

        Raises:
            RuntimeError: If downloading or loading the DAC model fails.
        """
        import dac

        try:
            dac_model_path = dac.utils.download()
            dac_model = dac.DAC.load(dac_model_path)
            dac_model.eval()
        except Exception as e:
            raise RuntimeError("Failed to load DAC model") from e
        self.dac_model = dac_model

    def _encode_text(self, text: str) -> jnp.ndarray:
        """Encodes the input text string into a tensor of token IDs using byte-level encoding.

        Special tokens [S1] and [S2] are replaced by their byte values. The resulting
        sequence is truncated to the maximum configured text length.

        Args:
            text: The input text string.

        Returns:
            A tensor containing the encoded byte token IDs.
        """
        max_len = self.config.encoder_config.max_position_embeddings

        byte_text = text.encode("utf-8")
        replaced_bytes = (
            byte_text.replace(b"[S1]", b"\x01")
            .replace(b"[S2]", b"\x02")
            .replace(b"[S3]", b"\x03")
            .replace(b"[S4]", b"\x04")
        )
        text_tokens = list(replaced_bytes)

        return jnp.array(text_tokens[:max_len], dtype=jnp.int32)

    def _pad_text_input(self, text_tokens: list[jnp.ndarray]) -> jnp.ndarray:
        """Pads the text input to the maximum length."""
        text_pad_value = 0
        max_len = self.config.encoder_config.max_position_embeddings
        batch_size = len(text_tokens)

        src_tokens = jnp.full(
            (batch_size, 1, max_len),
            fill_value=text_pad_value,
            dtype=jnp.int32,
        )

        for i in range(batch_size):
            current_len = len(text_tokens[i])
            src_tokens = src_tokens.at[i, 0, :current_len].set(text_tokens[i])

        return src_tokens

    def _prepare_audio_prompt(
        self, audio_prompts: list[jnp.ndarray | None]
    ) -> tuple[jnp.ndarray, list[int]]:
        """Prepares the audio prompt tensor for the decoder.

        Handles padding, adds the beginning-of-sequence (BOS) token, applies the
        delay pattern, and determines the number of prefill steps for each item
        in the batch.

        Args:
            audio_prompts: A list of audio prompt tensors (encoded DAC frames) or None.
                           Each tensor should have shape [T, C].

        Returns:
            A tuple containing:
                - delayed_batch (torch.Tensor): The prepared audio prompt tensor with
                  delays applied, shape [B, T_max_padded, C].
                - prefill_steps (list[int]): A list containing the number of valid
                  tokens (including BOS) for each prompt in the batch.
        """
        num_channels = self.config.decoder_config.num_channels
        audio_bos_value = self.config.bos_token_id
        delay_pattern = self.config.delay_pattern
        max_delay_pattern = max(delay_pattern)
        batch_size = len(audio_prompts)

        max_len = (
            max(p.shape[0] if p is not None else 0 for p in audio_prompts)
            + max_delay_pattern
        )
        prefill_steps = []

        prefill = jnp.full(
            (batch_size, max_len, num_channels),
            fill_value=-1,
            dtype=jnp.int32,
        )

        prefill = prefill.at[:, 0, :].set(audio_bos_value)

        for i in range(batch_size):
            prompt = audio_prompts[i]
            if prompt is not None:
                prompt = prompt.astype(jnp.int32)
                prefill = prefill.at[i, 1 : prompt.shape[0] + 1, :].set(prompt)
                prefill_steps.append(prompt.shape[0] + 1)
            else:
                prefill_steps.append(1)

        delay_precomp = build_delay_indices(
            B=batch_size,
            T=max_len,
            C=num_channels,
            delay_pattern=delay_pattern,
        )

        delayed_batch = apply_audio_delay(
            audio_BxTxC=prefill,
            pad_value=-1,
            bos_value=audio_bos_value,
            precomp=delay_precomp,
        )

        return delayed_batch, prefill_steps

    def _prepare_generation(
        self,
        text: jnp.ndarray,
        audio_prompts: list[jnp.ndarray | None],
        max_tokens: int | None = None,
    ):
        """Initializes the model state for generation.

        Encodes the text input (conditional and unconditional), prepares the
        encoder and decoder states (including KV caches and cross-attention),
        prepares the audio prompt, and performs the initial decoder prefill steps
        based on the audio prompts.

        Args:
            text: The padded text input tensor, shape [B, 1, T_text].
            audio_prompts: A list of prepared audio prompt tensors or None.

        Returns:
            A tuple containing:
                - dec_state (DecoderInferenceState): The initialized decoder state.
                - dec_output (DecoderOutput): The initialized decoder output manager,
                  containing the prefilled audio tokens.
        """
        batch_size = text.shape[0]

        enc_input_uncond = jnp.zeros_like(text)
        enc_input_cond = text
        stacked_inputs = jnp.stack([enc_input_uncond, enc_input_cond], axis=1)
        enc_input = stacked_inputs.reshape(2 * batch_size, -1)

        enc_state = EncoderInferenceState.new(self.config, enc_input_cond)
        encoder_out = self.model.encoder(enc_input, enc_state)

        dec_cross_attn_cache = self.model.decoder.precompute_cross_attn_cache(
            encoder_out
        )
        dec_state = DecoderInferenceState.new(
            self.config,
            enc_state,
            encoder_out,
            dec_cross_attn_cache,
            self.compute_dtype,
            max_generation_length=max_tokens,
        )

        prefill, prefill_steps = self._prepare_audio_prompt(audio_prompts)

        dec_output = DecoderOutput.new(batch_size, self.config)
        dec_output.prefill(prefill, prefill_steps)

        dec_step = min(prefill_steps) - 1
        if dec_step > 0:
            dec_state.prepare_step(0, dec_step)
            tokens_BxTxC = dec_output.get_tokens_at(0, dec_step)
            tokens_BxTxC = jnp.repeat(tokens_BxTxC, 2, axis=0)
            self.model.decoder(tokens_BxTxC, dec_state)

        return dec_state, dec_output

    def _decoder_step(
        self,
        tokens_Bx1xC: jnp.ndarray,
        dec_state: DecoderInferenceState,
        cfg_scale: float,
        temperature: float,
        top_p: float,
        top_k: int,
        current_idx: int,
        step_key: jax.Array,
    ) -> jnp.ndarray:
        """Performs a single step of the decoder inference.

        Takes the tokens from the previous step, runs them through the decoder
        (for both conditional and unconditional paths), applies classifier-free
        guidance (CFG), samples the next token using temperature, top-p, and top-k
        sampling, and applies constraints (e.g., preventing EOS in certain channels).

        Args:
            tokens_Bx1xC: The input tokens for the current step, shape [2*B, 1, C].
                         Repeated for CFG (unconditional and conditional).
            dec_state: The current state of the decoder (KV caches, etc.).
            cfg_scale: The scale factor for classifier-free guidance.
            temperature: The temperature for sampling.
            top_p: The cumulative probability threshold for top-p sampling.
            top_k: The number of top logits to consider for top-k sampling.
            current_idx: The current generation step index.

        Returns:
            torch.Tensor: The sampled next tokens for each item in the batch,
                          shape [B, C].
        """
        B = tokens_Bx1xC.shape[0] // 2

        audio_eos_value = self.config.eos_token_id
        logits_Bx1xCxV = self.model.decoder.decode_step(
            tokens_Bx1xC, dec_state, current_idx
        )

        logits_last_2BxCxV = logits_Bx1xCxV[:, -1]
        logits_last_Bx2xCxV = logits_last_2BxCxV.reshape(
            B, 2, *logits_last_2BxCxV.shape[1:]
        )

        uncond_logits_BxCxV = logits_last_Bx2xCxV[:, 0, :, :]
        cond_logits_BxCxV = logits_last_Bx2xCxV[:, 1, :, :]

        # CFG-filter: Use CFG logits to determine top-k mask, but apply to original conditional logits
        cfg_logits_BxCxV = cond_logits_BxCxV + cfg_scale * (
            cond_logits_BxCxV - uncond_logits_BxCxV
        )

        _, top_k_indices_BxCxk = jax.lax.top_k(cfg_logits_BxCxV, k=top_k)
        mask_BxCxV = jnp.ones_like(cond_logits_BxCxV, dtype=bool)
        batch_indices = jnp.arange(mask_BxCxV.shape[0])[:, None, None]
        channel_indices = jnp.arange(mask_BxCxV.shape[1])[None, :, None]
        mask_BxCxV = mask_BxCxV.at[
            batch_indices, channel_indices, top_k_indices_BxCxk
        ].set(False)
        # Apply mask to original conditional logits to avoid CFG speed-up bias
        logits_BxCxV = jnp.where(mask_BxCxV, -jnp.inf, cond_logits_BxCxV)

        logits_BxCxV = logits_BxCxV.at[:, :, audio_eos_value + 1 :].set(-jnp.inf)
        logits_BxCxV = logits_BxCxV.at[:, 1:, audio_eos_value:].set(-jnp.inf)

        flat_logits_BCxV = logits_BxCxV.reshape(
            B * self.config.decoder_config.num_channels, -1
        )

        pred_BC = _sample_next_token(
            flat_logits_BCxV,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            audio_eos_value=audio_eos_value,
            rng_key=rng_key,
        )

        pred_BxC = pred_BC.reshape(B, self.config.decoder_config.num_channels)
        return pred_BxC

    def _generate_output(
        self, generated_codes: jnp.ndarray, lengths_Bx: jnp.ndarray
    ) -> list[np.ndarray]:
        """Converts generated delayed codes into audio waveforms.

        Reverts the delay pattern applied during generation, decodes the resulting
        codebook using the DAC model (if loaded), and returns a list of audio
        waveforms as NumPy arrays. If DAC is not loaded, returns the raw codebook indices.

        Args:
            generated_codes: The tensor of generated audio codes with delays,
                             shape [B, T_gen, C].
            lengths_Bx: A tensor containing the valid length of generated codes
                        (excluding padding and BOS/EOS markers) for each item
                        in the batch, shape [B].

        Returns:
            A list of NumPy arrays, where each array represents the generated audio
            waveform for one item in the batch. If DAC is not loaded, returns the
            raw, reverted codebook indices as NumPy arrays.
        """
        num_channels = self.config.decoder_config.num_channels
        batch_size = generated_codes.shape[0]
        seq_length = generated_codes.shape[1]
        delay_pattern = self.config.delay_pattern
        audio_pad_value = self.config.pad_token_id
        max_delay_pattern = max(delay_pattern)

        revert_precomp = build_revert_indices(
            B=batch_size,
            T=seq_length,
            C=num_channels,
            delay_pattern=delay_pattern,
        )

        codebook = revert_audio_delay(
            audio_BxTxC=generated_codes,
            pad_value=audio_pad_value,
            precomp=revert_precomp,
            T=seq_length,
        )[:, :-max_delay_pattern, :]

        min_valid_index = 0
        max_valid_index = 1023
        invalid_mask = (codebook < min_valid_index) | (codebook > max_valid_index)
        codebook = jnp.where(invalid_mask, 0, codebook)

        audios = []

        if self.load_dac:
            import torch

            for i in range(batch_size):
                # Convert JAX array to torch tensor for DAC
                # We don't have a JAX version of the DAC model yet
                codes_torch = torch.from_numpy(
                    np.array(codebook[i, : lengths_Bx[i], :])
                )
                audio = self._decode(codes_torch)
                audio_np = audio.detach().cpu().numpy()
                audios.append(audio_np)
        else:
            for i in range(batch_size):
                audios.append(np.array(codebook[i, : lengths_Bx[i], :]))
        return audios

    def _encode(self, audio):
        """Encodes the given audio waveform into a tensor of DAC codebook indices"""
        audio = audio.unsqueeze(0)
        audio_data = self.dac_model.preprocess(audio, DEFAULT_SAMPLE_RATE)
        _, encoded_frame, _, _, _ = self.dac_model.encode(audio_data)
        return encoded_frame.squeeze(0).transpose(0, 1)

    def _decode(self, audio_codes):
        """Decodes the given frames into an output audio waveform"""
        audio_codes = audio_codes.unsqueeze(0).transpose(1, 2)
        audio_values, _, _ = self.dac_model.quantizer.from_codes(audio_codes)
        audio_values = self.dac_model.decode(audio_values)
        return audio_values.squeeze()

    def load_audio(self, audio_path: str) -> jnp.ndarray:
        """Loads and preprocesses an audio file for use as a prompt.

        Loads the audio file, resamples it to the target sample rate if necessary,
        preprocesses it using the DAC model's preprocessing, and encodes it into
        DAC codebook indices.

        Args:
            audio_path: Path to the audio file.

        Returns:
            torch.Tensor: The encoded audio prompt as DAC codebook indices,
                          shape [T, C].

        Raises:
            RuntimeError: If the DAC model is not loaded (`load_dac=False` during init).
            FileNotFoundError: If the audio file cannot be found.
            Exception: If there's an error during loading or processing.
        """
        if self.dac_model is None:
            raise RuntimeError(
                "DAC model is required for loading audio prompts but was not loaded."
            )

        import torchaudio
        import torch

        audio, sr = torchaudio.load(audio_path, channels_first=True)
        if sr != DEFAULT_SAMPLE_RATE:
            audio = torchaudio.functional.resample(audio, sr, DEFAULT_SAMPLE_RATE)
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)

        encoded = self._encode(audio)
        return jnp.array(encoded.cpu().numpy())

    def save_audio(self, path: str, audio: np.ndarray):
        """Saves the generated audio waveform to a file."""
        import soundfile as sf

        sf.write(path, audio, DEFAULT_SAMPLE_RATE)

    def get_model_stats(self) -> dict[str, int | float]:
        """Get model statistics including parameter count and size.

        Returns:
            Dictionary containing:
                - total_params: Total number of parameters
                - param_size_mb: Estimated model size in MB
        """
        _, state = nnx.split(self.model)
        state_dict = nnx.to_pure_dict(state)

        # Count parameters
        param_count = sum(x.size for x in jax.tree_util.tree_leaves(state_dict))

        # Calculate model size in MB (assuming float32)
        param_size_mb = param_count * 4 / (1024 * 1024)

        return {"total_params": param_count, "param_size_mb": param_size_mb}

    def generate(
        self,
        text: str | list[str],
        max_tokens: int = 3072,
        cfg_scale: float = 3.0,
        temperature: float = 1.2,
        top_p: float = 0.95,
        cfg_filter_top_k: int = 45,
        audio_prompt: list[str | jnp.ndarray | None] | str | jnp.ndarray | None = None,
        audio_prompt_path: list[str | jnp.ndarray | None]
        | str
        | jnp.ndarray
        | None = None,
        verbose: bool = False,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ) -> np.ndarray | list[np.ndarray]:
        """Generates audio corresponding to the input text.

        Args:
            text: The input text prompt, or a list of text prompts for batch generation.
            max_tokens: The maximum number of audio tokens to generate per prompt.
                        Defaults to the model's configured audio length if None.
            cfg_scale: The scale factor for classifier-free guidance (CFG). Higher values
                       lead to stronger guidance towards the text prompt.
            temperature: The temperature for sampling. Higher values increase randomness.
            top_p: The cumulative probability threshold for nucleus (top-p) sampling.
            use_torch_compile: Whether to compile the generation steps using torch.compile.
                               Can significantly speed up generation after the initial
                               compilation overhead. Defaults to False.
            cfg_filter_top_k: The number of top logits to consider during CFG filtering.
                              (Note: This parameter name might be slightly misleading based
                              on the code; it's used in the `_sample_next_token` function.)
            audio_prompt: An audio prompt or list of prompts to condition the generation.
                          Can be a file path (str), a pre-loaded tensor (DAC codes), or None.
                          If a list, its length must match the batch size of the text input.
            audio_prompt_path: (Deprecated) Use `audio_prompt` instead.
            verbose: If True, prints progress information during generation, including
                     speed metrics.

        Returns:
            If a single text prompt was provided, returns a NumPy array containing the
            generated audio waveform.
            If a list of text prompts was provided, returns a list of NumPy arrays,
            each corresponding to a prompt in the input list. Returns None for a
            sequence if no audio was generated for it.
        """
        rng_key = rngs()

        batch_size = len(text) if isinstance(text, list) else 1
        audio_eos_value = self.config.eos_token_id
        audio_pad_value = self.config.pad_token_id
        delay_pattern = self.config.delay_pattern
        max_delay_pattern = max(delay_pattern)
        delay_pattern_Cx = jnp.array(delay_pattern, dtype=jnp.int32)

        if audio_prompt_path:
            print("Warning: audio_prompt_path is deprecated. Use audio_prompt instead.")
            audio_prompt = audio_prompt_path

        if verbose:
            total_start_time = time.time()

        if isinstance(audio_prompt, list):
            audio_prompt = [
                self.load_audio(p) if isinstance(p, str) else p for p in audio_prompt
            ]
        elif isinstance(audio_prompt, str):
            audio_prompt = [self.load_audio(audio_prompt)]
        elif isinstance(audio_prompt, jnp.ndarray):
            audio_prompt = [audio_prompt]
        elif audio_prompt is None:
            audio_prompt = [None] * batch_size

        assert len(audio_prompt) == batch_size, (
            "Number of audio prompts must match batch size"
        )

        if isinstance(text, list):
            text = [self._encode_text(t) for t in text]
        else:
            text = [self._encode_text(text)]
        text = self._pad_text_input(text)

        dec_state, dec_output = self._prepare_generation(
            text, audio_prompt, max_tokens=max_tokens
        )
        dec_step = min(dec_output.prefill_steps) - 1
        current_idx = dec_step

        eos_detected_Bx = jnp.zeros((batch_size,), dtype=bool)
        eos_countdown_Bx = jnp.full((batch_size,), -1, dtype=jnp.int32)
        finished_step_Bx = jnp.full((batch_size,), -1, dtype=jnp.int32)

        bos_over = False

        if verbose:
            print("generate: starting generation loop")
            start_time = time.time()

        # Generation Loop
        while dec_step < max_tokens:
            if jnp.all(eos_countdown_Bx == 0):
                break

            current_step_idx = dec_step + 1
            dec_state.prepare_step(dec_step)
            tokens_Bx1xC = dec_output.get_tokens_at(dec_step)
            tokens_Bx1xC = jnp.repeat(tokens_Bx1xC, 2, axis=0)

            rng_key, step_key = jax.random.split(rng_key)
            pred_BxC = self._decoder_step(
                tokens_Bx1xC,
                dec_state,
                cfg_scale,
                temperature,
                top_p,
                cfg_filter_top_k,
                current_idx,
                step_key,
            )

            current_idx += 1

            active_mask_Bx = eos_countdown_Bx != 0
            eos_trigger_Bx = jnp.zeros_like(active_mask_Bx)

            if jnp.any(active_mask_Bx):
                is_eos_token = (~eos_detected_Bx[active_mask_Bx]) & (
                    pred_BxC[active_mask_Bx, 0] == audio_eos_value
                )
                is_max_len = current_step_idx >= max_tokens - max_delay_pattern
                eos_trigger_Bx = eos_trigger_Bx.at[active_mask_Bx].set(
                    is_eos_token | is_max_len
                )

            eos_detected_Bx = eos_detected_Bx | eos_trigger_Bx
            start_countdown_mask_Bx = eos_trigger_Bx & (eos_countdown_Bx < 0)

            if jnp.any(start_countdown_mask_Bx):
                eos_countdown_Bx = eos_countdown_Bx.at[start_countdown_mask_Bx].set(
                    max_delay_pattern
                )
                finished_step_Bx = finished_step_Bx.at[start_countdown_mask_Bx].set(
                    current_step_idx
                )

            padding_mask_Bx = eos_countdown_Bx > 0
            if jnp.any(padding_mask_Bx):
                pred_active_BxC = pred_BxC[padding_mask_Bx]
                countdown_active_Bx = eos_countdown_Bx[padding_mask_Bx]
                step_after_eos_Bx = max_delay_pattern - countdown_active_Bx
                step_after_eos_Bx_ = step_after_eos_Bx[:, None]
                delay_pattern_Cx_ = delay_pattern_Cx[None, :]
                eos_mask_NxC = step_after_eos_Bx_ == delay_pattern_Cx_
                pad_mask_NxC = step_after_eos_Bx_ > delay_pattern_Cx_
                pred_active_BxC = jnp.where(
                    eos_mask_NxC, audio_eos_value, pred_active_BxC
                )
                pred_active_BxC = jnp.where(
                    pad_mask_NxC, audio_pad_value, pred_active_BxC
                )
                pred_BxC = pred_BxC.at[padding_mask_Bx].set(pred_active_BxC)
                eos_countdown_Bx = eos_countdown_Bx.at[padding_mask_Bx].add(-1)

            if not bos_over:
                bos_over = all(
                    dec_step - prefill_step > max_delay_pattern
                    for prefill_step in dec_output.prefill_steps
                )

            dec_output.update_one(pred_BxC, current_step_idx, not bos_over)

            dec_step += 1

            if verbose and dec_step % 86 == 0:
                duration = time.time() - start_time
                if duration > 0:
                    print(
                        f"generate step {dec_step}: speed={86 * batch_size / duration:.3f} tokens/s, realtime factor={batch_size / duration:.3f}x"
                    )
                start_time = time.time()

        # Finalize and Extract Output
        final_step = dec_step + 1

        finished_step_Bx = jnp.where(
            finished_step_Bx == -1, final_step - max_delay_pattern, finished_step_Bx
        )

        prefill_steps_tensor = jnp.array(dec_output.prefill_steps)
        lengths_Bx = finished_step_Bx - prefill_steps_tensor
        lengths_Bx = jnp.clip(lengths_Bx, a_min=0)

        max_len = int(lengths_Bx.max()) + max_delay_pattern
        outputs = []

        if max_len > 0:
            num_channels = self.config.decoder_config.num_channels
            generated_codes = jnp.full(
                (batch_size, max_len, num_channels),
                fill_value=audio_pad_value,
                dtype=jnp.int32,
            )

            for i in range(batch_size):
                start_step = dec_output.prefill_steps[i]
                actual_len = int(lengths_Bx[i]) + max_delay_pattern
                if actual_len > 0:
                    tokens_to_copy = dec_output.generated_tokens[
                        i, start_step : start_step + actual_len, :
                    ]
                    generated_codes = generated_codes.at[i, :actual_len, :].set(
                        tokens_to_copy
                    )

            if verbose:
                avg_steps = float(lengths_Bx.mean())
                total_duration = time.time() - total_start_time
                print(
                    f"generate: avg steps={avg_steps:.1f}, total duration={total_duration:.3f}s"
                )

            outputs = self._generate_output(generated_codes, lengths_Bx)
        else:
            print("Warning: Nothing generated for any sequence in the batch.")
            outputs = [None] * batch_size

        return outputs if batch_size > 1 else outputs[0]

    def generate_jit(
        self,
        text: str | list[str],
        audio_prompt: list[str | jnp.ndarray | None] | str | jnp.ndarray | None = None,
        max_tokens: int = 3072,
        cfg_scale: float = 3.0,
        temperature: float = 1.2,
        top_p: float = 0.95,
        cfg_filter_top_k: int = 45,
        verbose: bool = False,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ):
        """Hot-path JIT compiled for efficient sampling"""

        rng_key = rngs()

        batch_size = len(text) if isinstance(text, list) else 1
        audio_eos_value = self.config.eos_token_id
        audio_pad_value = self.config.pad_token_id
        delay_pattern = self.config.delay_pattern
        max_delay_pattern = max(delay_pattern)
        delay_pattern_Cx = jnp.array(delay_pattern, dtype=jnp.int32)

        if verbose:
            total_start_time = time.time()

        if isinstance(audio_prompt, list):
            audio_prompt = [
                self.load_audio(p) if isinstance(p, str) else p for p in audio_prompt
            ]
        elif isinstance(audio_prompt, str):
            audio_prompt = [self.load_audio(audio_prompt)]
        elif isinstance(audio_prompt, jnp.ndarray):
            audio_prompt = [audio_prompt]
        elif audio_prompt is None:
            audio_prompt = [None] * batch_size

        assert len(audio_prompt) == batch_size, (
            "Number of audio prompts must match batch size"
        )

        if isinstance(text, list):
            text = [self._encode_text(t) for t in text]
        else:
            text = [self._encode_text(text)]
        text = self._pad_text_input(text)

        dec_state, dec_output = self._prepare_generation(
            text, audio_prompt, max_tokens=max_tokens
        )
        dec_step = min(dec_output.prefill_steps) - 1
        current_idx = dec_step

        eos_detected_Bx = jnp.zeros((batch_size,), dtype=bool)
        eos_countdown_Bx = jnp.full((batch_size,), -1, dtype=jnp.int32)
        finished_step_Bx = jnp.full((batch_size,), -1, dtype=jnp.int32)

        bos_over = False

        if verbose:
            print("generate: starting generation loop")
            start_time = time.time()

        @nnx.jit(
            static_argnames=["dia_config", "cfg_scale", "top_k", "temperature", "top_p"]
        )
        def decode_step(
            model: DiaModel,
            dia_config: DiaConfig,
            tokens_Bx1xC: jnp.ndarray,
            current_idx: int,
            causal_attn_mask: jnp.ndarray,
            cross_attn_mask: jnp.ndarray,
            dec_positions: jnp.ndarray,
            enc_positions: jnp.ndarray,
            enc_out: jnp.ndarray,
            self_attn_cache: list[KVCache],
            cross_attn_cache: list[KVCache],
            step_key: nnx.RngKey,
            cfg_scale: float = 3.0,
            top_k: int = 50,
            temperature: float = 1.2,
            top_p: float = 0.95,
        ):
            """Single JIT compiled step"""
            tokens_Bx1xC = jnp.repeat(tokens_Bx1xC, 2, axis=0)

            B = tokens_Bx1xC.shape[0] // 2
            audio_eos_value = dia_config.eos_token_id
            logits_Bx1xCxV, self_attn_cache, cross_attn_cache = (
                model.decoder.decode_step_jit(
                    tokens_Bx1xC,
                    current_idx,
                    causal_attn_mask,
                    cross_attn_mask,
                    dec_positions,
                    enc_positions,
                    enc_out,
                    self_attn_cache,
                    cross_attn_cache,
                )
            )
            logits_last_2BxCxV = logits_Bx1xCxV[:, -1]
            logits_last_Bx2xCxV = logits_last_2BxCxV.reshape(
                B, 2, *logits_last_2BxCxV.shape[1:]
            )

            uncond_logits_BxCxV = logits_last_Bx2xCxV[:, 0, :, :]
            cond_logits_BxCxV = logits_last_Bx2xCxV[:, 1, :, :]

            # CFG-filter: Use CFG logits to determine top-k mask, but apply to original conditional logits
            cfg_logits_BxCxV = cond_logits_BxCxV + cfg_scale * (
                cond_logits_BxCxV - uncond_logits_BxCxV
            )

            _, top_k_indices_BxCxk = jax.lax.top_k(cfg_logits_BxCxV, k=top_k)
            mask_BxCxV = jnp.ones_like(cond_logits_BxCxV, dtype=bool)
            batch_indices = jnp.arange(mask_BxCxV.shape[0])[:, None, None]
            channel_indices = jnp.arange(mask_BxCxV.shape[1])[None, :, None]
            mask_BxCxV = mask_BxCxV.at[
                batch_indices, channel_indices, top_k_indices_BxCxk
            ].set(False)

            # Apply mask to original conditional logits to avoid CFG speed-up bias
            logits_BxCxV = jnp.where(mask_BxCxV, -jnp.inf, cond_logits_BxCxV)

            logits_BxCxV = logits_BxCxV.at[:, :, audio_eos_value + 1 :].set(-jnp.inf)
            logits_BxCxV = logits_BxCxV.at[:, 1:, audio_eos_value:].set(-jnp.inf)

            flat_logits_BCxV = logits_BxCxV.reshape(
                B * dia_config.decoder_config.num_channels, -1
            )

            pred_BC = _sample_next_token(
                flat_logits_BCxV,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                audio_eos_value=audio_eos_value,
                step_key=step_key,
            )
            pred_BxC = pred_BC.reshape(B, dia_config.decoder_config.num_channels)
            return pred_BxC, self_attn_cache, cross_attn_cache

        while dec_step < max_tokens:
            if jnp.all(eos_countdown_Bx == 0):
                break

            current_step_idx = dec_step + 1
            dec_state.prepare_step(dec_step)
            tokens_Bx1xC = dec_output.get_tokens_at(dec_step)
            rng_key, step_key = jax.random.split(rng_key)
            pred_BxC, self_attn_cache, cross_attn_cache = decode_step(
                self.model,
                self.config,
                tokens_Bx1xC,
                current_idx,
                dec_state.causal_attn_mask,
                dec_state.cross_attn_mask,
                dec_state.dec_positions,
                dec_state.enc_positions,
                dec_state.enc_out,
                dec_state.self_attn_cache,
                dec_state.cross_attn_cache,
                step_key,
                cfg_scale,
                cfg_filter_top_k,
                temperature,
                top_p,
            )

            current_idx += 1

            # Update the decoder state caches with the new values from JIT
            dec_state.self_attn_cache = self_attn_cache
            dec_state.cross_attn_cache = cross_attn_cache

            active_mask_Bx = eos_countdown_Bx != 0
            eos_trigger_Bx = jnp.zeros_like(active_mask_Bx)

            if jnp.any(active_mask_Bx):
                is_eos_token = (~eos_detected_Bx[active_mask_Bx]) & (
                    pred_BxC[active_mask_Bx, 0] == audio_eos_value
                )
                is_max_len = current_step_idx >= max_tokens - max_delay_pattern
                eos_trigger_Bx = eos_trigger_Bx.at[active_mask_Bx].set(
                    is_eos_token | is_max_len
                )

            eos_detected_Bx = eos_detected_Bx | eos_trigger_Bx
            start_countdown_mask_Bx = eos_trigger_Bx & (eos_countdown_Bx < 0)

            if jnp.any(start_countdown_mask_Bx):
                eos_countdown_Bx = eos_countdown_Bx.at[start_countdown_mask_Bx].set(
                    max_delay_pattern
                )
                finished_step_Bx = finished_step_Bx.at[start_countdown_mask_Bx].set(
                    current_step_idx
                )

            padding_mask_Bx = eos_countdown_Bx > 0
            if jnp.any(padding_mask_Bx):
                pred_active_BxC = pred_BxC[padding_mask_Bx]
                countdown_active_Bx = eos_countdown_Bx[padding_mask_Bx]
                step_after_eos_Bx = max_delay_pattern - countdown_active_Bx
                step_after_eos_Bx_ = step_after_eos_Bx[:, None]
                delay_pattern_Cx_ = delay_pattern_Cx[None, :]
                eos_mask_NxC = step_after_eos_Bx_ == delay_pattern_Cx_
                pad_mask_NxC = step_after_eos_Bx_ > delay_pattern_Cx_
                pred_active_BxC = jnp.where(
                    eos_mask_NxC, audio_eos_value, pred_active_BxC
                )
                pred_active_BxC = jnp.where(
                    pad_mask_NxC, audio_pad_value, pred_active_BxC
                )
                pred_BxC = pred_BxC.at[padding_mask_Bx].set(pred_active_BxC)
                eos_countdown_Bx = eos_countdown_Bx.at[padding_mask_Bx].add(-1)

            if not bos_over:
                bos_over = all(
                    dec_step - prefill_step > max_delay_pattern
                    for prefill_step in dec_output.prefill_steps
                )

            dec_output.update_one(pred_BxC, current_step_idx, not bos_over)

            dec_step += 1

            if verbose and dec_step % 86 == 0:
                duration = time.time() - start_time
                if duration > 0:
                    print(
                        f"generate step {dec_step}: speed={86 * batch_size / duration:.3f} tokens/s, realtime factor={batch_size / duration:.3f}x"
                    )
                start_time = time.time()

        # Finalize and Extract Output
        final_step = dec_step + 1

        finished_step_Bx = jnp.where(
            finished_step_Bx == -1, final_step - max_delay_pattern, finished_step_Bx
        )

        prefill_steps_tensor = jnp.array(dec_output.prefill_steps)
        lengths_Bx = finished_step_Bx - prefill_steps_tensor
        lengths_Bx = jnp.clip(lengths_Bx, a_min=0)

        max_len = int(lengths_Bx.max()) + max_delay_pattern
        outputs = []

        if max_len > 0:
            num_channels = self.config.decoder_config.num_channels
            generated_codes = jnp.full(
                (batch_size, max_len, num_channels),
                fill_value=audio_pad_value,
                dtype=jnp.int32,
            )

            for i in range(batch_size):
                start_step = dec_output.prefill_steps[i]
                actual_len = int(lengths_Bx[i]) + max_delay_pattern
                if actual_len > 0:
                    tokens_to_copy = dec_output.generated_tokens[
                        i, start_step : start_step + actual_len, :
                    ]
                    generated_codes = generated_codes.at[i, :actual_len, :].set(
                        tokens_to_copy
                    )

            if verbose:
                avg_steps = float(lengths_Bx.mean())
                total_duration = time.time() - total_start_time
                print(
                    f"generate: avg steps={avg_steps:.1f}, total duration={total_duration:.3f}s"
                )

            outputs = self._generate_output(generated_codes, lengths_Bx)
        else:
            print("Warning: Nothing generated for any sequence in the batch.")
            outputs = [None] * batch_size

        return outputs if batch_size > 1 else outputs[0]
