import jax.numpy as jnp


def build_delay_indices(B: int, T: int, C: int, delay_pattern: list[int]):
    """
    Precompute (t_idx_BxTxC, indices_BTCx3) so that out[t, c] = in[t - delay[c], c].
    Negative t_idx => BOS; t_idx >= T => PAD.
    """
    delay_arr = jnp.array(delay_pattern, dtype=jnp.int32)

    t_idx_BxT = jnp.broadcast_to(jnp.arange(T, dtype=jnp.int32)[None, :], (B, T))
    t_idx_BxTx1 = t_idx_BxT[..., None]
    t_idx_BxTxC = t_idx_BxTx1 - delay_arr.reshape(1, 1, C)

    b_idx_BxTxC = jnp.broadcast_to(
        jnp.arange(B, dtype=jnp.int32).reshape(B, 1, 1), (B, T, C)
    )
    c_idx_BxTxC = jnp.broadcast_to(
        jnp.arange(C, dtype=jnp.int32).reshape(1, 1, C), (B, T, C)
    )

    t_clamped_BxTxC = jnp.clip(t_idx_BxTxC, 0, T - 1)

    indices_BTCx3 = jnp.stack(
        [
            b_idx_BxTxC.reshape(-1),
            t_clamped_BxTxC.reshape(-1),
            c_idx_BxTxC.reshape(-1),
        ],
        axis=1,
    )

    return t_idx_BxTxC, indices_BTCx3


def apply_audio_delay(
    audio_BxTxC: jnp.ndarray,
    pad_value: int,
    bos_value: int,
    precomp: tuple[jnp.ndarray, jnp.ndarray],
) -> jnp.ndarray:
    """
    Applies the delay pattern to batched audio tokens using precomputed indices,
    inserting BOS where t_idx < 0 and PAD where t_idx >= T.

    Args:
        audio_BxTxC: [B, T, C] int16 audio tokens (or int32/float)
        pad_value: the padding token
        bos_value: the BOS token
        precomp:  (t_idx_BxTxC, indices_BTCx3) from build_delay_indices

    Returns:
        result_BxTxC: [B, T, C] delayed audio tokens
    """
    t_idx_BxTxC, indices_BTCx3 = precomp

    gathered_flat = audio_BxTxC[
        indices_BTCx3[:, 0], indices_BTCx3[:, 1], indices_BTCx3[:, 2]
    ]
    gathered_BxTxC = gathered_flat.reshape(audio_BxTxC.shape)

    mask_bos = t_idx_BxTxC < 0
    mask_pad = t_idx_BxTxC >= audio_BxTxC.shape[1]

    result_BxTxC = jnp.where(
        mask_bos, bos_value, jnp.where(mask_pad, pad_value, gathered_BxTxC)
    )

    return result_BxTxC


def build_revert_indices(B: int, T: int, C: int, delay_pattern: list[int]):
    """
    Precompute indices for the revert operation using JAX.

    Returns:
        A tuple (t_idx_unclamped_BxTxC, t_idx_clamped_BxTxC, indices_BTCx3) where:
            - t_idx_unclamped_BxTxC: unclamped time indices (t + delay[c]) for padding check
            - t_idx_clamped_BxTxC: clamped time indices for safe gathering
            - indices_BTCx3: gathering indices using clamped time indices
    """
    delay_arr = jnp.array(delay_pattern, dtype=jnp.int32)

    t_idx_BT1 = jnp.broadcast_to(jnp.arange(T).reshape(1, T, 1), (B, T, 1))

    # Unclamped time indices for padding check
    t_idx_unclamped_BxTxC = t_idx_BT1 + delay_arr.reshape(1, 1, C)

    # Clamped time indices for safe gathering
    t_idx_clamped_BxTxC = jnp.minimum(
        t_idx_unclamped_BxTxC,
        T - 1,
    )

    b_idx_BxTxC = jnp.broadcast_to(jnp.arange(B).reshape(B, 1, 1), (B, T, C))
    c_idx_BxTxC = jnp.broadcast_to(jnp.arange(C).reshape(1, 1, C), (B, T, C))

    indices_BTCx3 = jnp.stack(
        [
            b_idx_BxTxC.reshape(-1),
            t_idx_clamped_BxTxC.reshape(-1),
            c_idx_BxTxC.reshape(-1),
        ],
        axis=1,
    )

    return t_idx_unclamped_BxTxC, t_idx_clamped_BxTxC, indices_BTCx3


def revert_audio_delay(
    audio_BxTxC: jnp.ndarray,
    pad_value: int,
    precomp: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    T: int,
) -> jnp.ndarray:
    """
    Reverts a delay pattern from batched audio tokens using precomputed indices (JAX version).

    Args:
        audio_BxTxC: Input delayed audio tensor
        pad_value: Padding value for out-of-bounds indices
        precomp: Precomputed revert indices tuple containing:
            - t_idx_unclamped_BxTxC: Unclamped time indices for padding check
            - t_idx_clamped_BxTxC: Clamped time indices (unused here)
            - indices_BTCx3: Gather indices tensor for original audio
        T: Original sequence length before padding

    Returns:
        Reverted audio tensor with same shape as input
    """
    t_idx_unclamped_BxTxC, t_idx_clamped_BxTxC, indices_BTCx3 = precomp

    gathered_flat = audio_BxTxC[
        indices_BTCx3[:, 0], indices_BTCx3[:, 1], indices_BTCx3[:, 2]
    ]
    gathered_BxTxC = gathered_flat.reshape(audio_BxTxC.shape)

    # Use unclamped indices to check for padding condition
    result_BxTxC = jnp.where(t_idx_unclamped_BxTxC >= T, pad_value, gathered_BxTxC)

    return result_BxTxC
