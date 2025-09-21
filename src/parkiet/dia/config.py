"""Configuration management module for the Dia model.

This module provides comprehensive configuration management for the Dia model,
utilizing Pydantic for validation. It defines configurations for data processing,
model architecture (encoder and decoder), and training settings.

Key components:
- DataConfig: Parameters for data loading and preprocessing.
- EncoderConfig: Architecture details for the encoder module.
- DecoderConfig: Architecture details for the decoder module.
- ModelConfig: Combined model architecture settings.
- TrainingConfig: Training hyperparameters and settings.
- DiaConfig: Master configuration combining all components.
"""

import os

from pydantic import BaseModel, Field, ConfigDict


class EncoderConfig(BaseModel):
    """Configuration for the encoder component of the Dia model.

    Attributes:
        model_type: Type of the model, defaults to "dia_encoder".
        hidden_size: Size of the encoder layers, defaults to 1024.
        intermediate_size: Size of the "intermediate" (i.e., feed-forward) layer in the encoder, defaults to 4096.
        num_hidden_layers: Number of hidden layers in the encoder, defaults to 12.
        num_attention_heads: Number of attention heads in the encoder, defaults to 16.
        num_key_value_heads: Number of key-value heads in the encoder, defaults to 16.
        head_dim: Dimension of each attention head, defaults to 128.
        hidden_act: Activation function in the encoder, defaults to "silu".
        max_position_embeddings: Maximum number of position embeddings, defaults to 1024.
        initializer_range: Range for initializing weights, defaults to 0.02.
        norm_eps: Epsilon value for normalization layers, defaults to 1e-5.
        rope_theta: Theta value for RoPE, defaults to 10000.0.
        rope_scaling: Optional scaling factor for RoPE.
        vocab_size: Vocabulary size, defaults to 256.
    """

    head_dim: int = Field(default=128, gt=0)
    hidden_act: str = Field(default="silu")
    hidden_size: int = Field(default=1024, gt=0)
    initializer_range: float = Field(default=0.02)
    intermediate_size: int = Field(default=4096, gt=0)
    max_position_embeddings: int = Field(default=1024, gt=0)
    model_type: str = Field(default="dia_encoder")
    norm_eps: float = Field(default=1e-5)
    num_attention_heads: int = Field(default=16, gt=0)
    num_hidden_layers: int = Field(default=12, gt=0)
    num_key_value_heads: int = Field(default=16, gt=0)
    rope_scaling: float | None = Field(default=None)
    rope_theta: float = Field(default=10000.0)
    vocab_size: int = Field(default=256, gt=0)

    model_config = ConfigDict(frozen=True)


class DecoderConfig(BaseModel):
    """Configuration for the decoder component of the Dia model.

    Attributes:
        model_type: Type of the model, defaults to "dia_decoder".
        hidden_size: Size of the decoder layers, defaults to 2048.
        intermediate_size: Size of the "intermediate" (i.e., feed-forward) layer in the decoder, defaults to 8192.
        num_hidden_layers: Number of hidden layers in the decoder, defaults to 18.
        num_attention_heads: Number of attention heads in the decoder, defaults to 16.
        num_key_value_heads: Number of key-value heads in the decoder, defaults to 4.
        head_dim: Dimension of each attention head, defaults to 128.
        cross_hidden_size: Size of the cross-attention layers, defaults to 1024.
        cross_num_attention_heads: Number of attention heads in the cross-attention mechanism, defaults to 16.
        cross_num_key_value_heads: Number of key-value heads in the cross-attention mechanism, defaults to 16.
        cross_head_dim: Dimension of each cross-attention head, defaults to 128.
        hidden_act: Activation function in the decoder, defaults to "silu".
        max_position_embeddings: Maximum number of position embeddings in the decoder, defaults to 3072.
        initializer_range: Range for initializing weights in the decoder, defaults to 0.02.
        norm_eps: Epsilon value for normalization layers in the decoder, defaults to 1e-5.
        rope_theta: Theta value for RoPE in the decoder, defaults to 10000.0.
        rope_scaling: Optional scaling factor for RoPE in the decoder.
        vocab_size: Vocabulary size for the decoder, defaults to 1028.
        num_channels: Number of channels in the decoder, defaults to 9.
    """

    cross_head_dim: int = Field(default=128, gt=0)
    cross_hidden_size: int = Field(default=1024, gt=0)
    cross_num_attention_heads: int = Field(default=16, gt=0)
    cross_num_key_value_heads: int = Field(default=16, gt=0)
    head_dim: int = Field(default=128, gt=0)
    hidden_act: str = Field(default="silu")
    hidden_size: int = Field(default=2048, gt=0)
    initializer_range: float = Field(default=0.02)
    intermediate_size: int = Field(default=8192, gt=0)
    max_position_embeddings: int = Field(default=3072, gt=0)
    model_type: str = Field(default="dia_decoder")
    norm_eps: float = Field(default=1e-5)
    num_attention_heads: int = Field(default=16, gt=0)
    num_channels: int = Field(default=9, gt=0)
    num_hidden_layers: int = Field(default=18, gt=0)
    num_key_value_heads: int = Field(default=4, gt=0)
    rope_scaling: float | None = Field(default=None)
    rope_theta: float = Field(default=10000.0)
    vocab_size: int = Field(default=1028, gt=0)

    model_config = ConfigDict(frozen=True)


class DiaConfig(BaseModel):
    """Main configuration container for the Dia model architecture.

    Attributes:
        model_type: Type of the model, defaults to "dia".
        is_encoder_decoder: Flag indicating if the model is an encoder-decoder type, defaults to True.
        encoder: Configuration for the encoder component.
        decoder: Configuration for the decoder component.
        src_vocab_size: Size of the source (text) vocabulary.
        tgt_vocab_size: Size of the target (audio code) vocabulary.
        initializer_range: Range for initializing weights, defaults to 0.02.
        norm_eps: Epsilon value for normalization layers, defaults to 1e-5.
        torch_dtype: Data type for model weights in PyTorch, defaults to "float32".
        bos_token_id: Beginning-of-sequence token ID, defaults to 1026.
        eos_token_id: End-of-sequence token ID, defaults to 1024.
        pad_token_id: Padding token ID, defaults to 1025.
        rope_theta: Theta value for RoPE, defaults to 10000.0.
        rope_scaling: Optional scaling factor for RoPE.
        transformers_version: Version of the transformers library, defaults to "4.53.0.dev0".
        architectures: List of model architectures, defaults to ["DiaForConditionalGeneration"].
        delay_pattern: List of delay values for each audio channel, defaults to [0,8,9,10,11,12,13,14,15].
    """

    architectures: tuple[str, ...] = Field(
        default_factory=lambda: ["DiaForConditionalGeneration"]
    )
    bos_token_id: int = Field(default=1026)
    decoder_config: DecoderConfig
    delay_pattern: tuple[int, ...] = Field(
        default_factory=lambda: (0, 8, 9, 10, 11, 12, 13, 14, 15)
    )
    encoder_config: EncoderConfig
    eos_token_id: int = Field(default=1024)
    initializer_range: float = Field(default=0.02)
    is_encoder_decoder: bool = Field(default=True)
    model_type: str = Field(default="dia")
    norm_eps: float = Field(default=1e-5)
    pad_token_id: int = Field(default=1025)
    torch_dtype: str = Field(default="float32")
    transformers_version: str = Field(default="4.53.0.dev0")

    model_config = ConfigDict(frozen=True)

    def save(self, path: str) -> None:
        """Save the current configuration instance to a JSON file.

        Ensures the parent directory exists and the file has a .json extension.

        Args:
            path: The target file path to save the configuration.

        Raises:
            ValueError: If the path is not a file with a .json extension.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        config_json = self.model_dump_json(indent=2)
        with open(path, "w") as f:
            f.write(config_json)

    @classmethod
    def load(cls, path: str) -> "DiaConfig | None":
        """Load and validate a Dia configuration from a JSON file.

        Args:
            path: The path to the configuration file.

        Returns:
            A validated DiaConfig instance if the file exists and is valid,
            otherwise None if the file is not found.

        Raises:
            ValueError: If the path does not point to an existing .json file.
            pydantic.ValidationError: If the JSON content fails validation against the DiaConfig schema.
        """
        try:
            with open(path, "r") as f:
                content = f.read()
            return cls.model_validate_json(content)
        except FileNotFoundError:
            return None
