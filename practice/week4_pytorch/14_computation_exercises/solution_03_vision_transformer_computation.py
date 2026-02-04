"""Solution: Vision Transformer 2D and 3D patch/sequence shapes and parameter counts."""

from typing import Tuple


def _to_pair(x: int | Tuple[int, int]) -> Tuple[int, int]:
    if isinstance(x, int):
        return (x, x)
    return x


def _to_triple(x: int | Tuple[int, int, int]) -> Tuple[int, int, int]:
    if isinstance(x, int):
        return (x, x, x)
    return x


def vit2d_num_patches(H: int, W: int, patch_size: int | Tuple[int, int]) -> int:
    Ph, Pw = _to_pair(patch_size)
    return (H // Ph) * (W // Pw)


def vit2d_sequence_length(H: int, W: int, patch_size: int | Tuple[int, int]) -> int:
    return vit2d_num_patches(H, W, patch_size) + 1


def vit2d_patch_embed_output_shape(
    B: int, C: int, H: int, W: int, patch_size: int | Tuple[int, int], d_model: int
) -> Tuple[int, int, int]:
    L = vit2d_sequence_length(H, W, patch_size)
    return (B, L, d_model)


def vit2d_patch_embed_num_params(
    C: int, d_model: int, patch_size: int | Tuple[int, int], bias: bool = True
) -> int:
    Ph, Pw = _to_pair(patch_size)
    return C * d_model * Ph * Pw + (d_model if bias else 0)


def vit3d_num_patches(
    T: int, H: int, W: int,
    patch_size: int | Tuple[int, int, int],
) -> int:
    Pt, Ph, Pw = _to_triple(patch_size)
    return (T // Pt) * (H // Ph) * (W // Pw)


def vit3d_sequence_length(
    T: int, H: int, W: int,
    patch_size: int | Tuple[int, int, int],
) -> int:
    return vit3d_num_patches(T, H, W, patch_size) + 1


def vit3d_patch_embed_output_shape(
    B: int, C: int, T: int, H: int, W: int,
    patch_size: int | Tuple[int, int, int],
    d_model: int,
) -> Tuple[int, int, int]:
    L = vit3d_sequence_length(T, H, W, patch_size)
    return (B, L, d_model)


def vit3d_patch_embed_num_params(
    C: int, d_model: int, patch_size: int | Tuple[int, int, int], bias: bool = True
) -> int:
    Pt, Ph, Pw = _to_triple(patch_size)
    return C * d_model * Pt * Ph * Pw + (d_model if bias else 0)


def vit2d_encoder_params(
    num_layers: int,
    d_model: int,
    num_heads: int,
    d_ff: int,
    patch_embed_params: int,
) -> int:
    # Per block: MHA (4*d_model^2 + 4*d_model) + FFN (d_model*d_ff + d_ff + d_ff*d_model + d_model) + 2*LayerNorm (4*d_model)
    mha = 4 * d_model * d_model + 4 * d_model
    ffn = d_model * d_ff + d_ff + d_ff * d_model + d_model
    ln = 4 * d_model
    per_block = mha + ffn + ln
    final_ln = 2 * d_model
    return patch_embed_params + num_layers * per_block + final_ln
