"""
14 — Vision Transformer (ViT) computation (2D image, 3D video)
================================================================
Learning goal: Patch embedding dimensions, sequence length, parameter counts for ViT
in both image (2D) and video (3D) settings. Explicit (B, C, H, W) and (B, C, T, H, W).

ViT 2D (image):
  - Input (B, C, H, W). Patch size P → grid (H/P) x (W/P). Num patches = (H/P)*(W/P).
  - Add [CLS] token → sequence length L = num_patches + 1.
  - Patch embedding: Conv2d(C, d_model, kernel=P, stride=P) → (B, d_model, H/P, W/P) → flatten → (B, d_model, L-1) → transpose → (B, L-1, d_model); then prepend [CLS] → (B, L, d_model).

ViT 3D (video):
  - Input (B, C, T, H, W). Patch size (P_t, P_h, P_w) → num patches = (T/P_t)*(H/P_h)*(W/P_w).
  - L = num_patches + 1. Rest same idea with Conv3d or 3D patch embed.

Implement the functions below. All dimensions explicit.
See: Computation_Exercises_Guide.md (Vision Transformers section).
"""

from typing import Tuple


# ---------------------------------------------------------------------------
# 2D ViT (image): input (B, C, H, W)
# ---------------------------------------------------------------------------


def vit2d_num_patches(H: int, W: int, patch_size: int | Tuple[int, int]) -> int:
    """
    Number of patches (excluding [CLS]) for a 2D image.
    patch_size can be int (P x P) or (P_h, P_w).
    """
    # TODO: return (H / P_h) * (W / P_w) as integer
    raise NotImplementedError


def vit2d_sequence_length(H: int, W: int, patch_size: int | Tuple[int, int]) -> int:
    """Sequence length after adding [CLS]: num_patches + 1."""
    # TODO: return vit2d_num_patches(...) + 1
    raise NotImplementedError


def vit2d_patch_embed_output_shape(
    B: int, C: int, H: int, W: int, patch_size: int | Tuple[int, int], d_model: int
) -> Tuple[int, int, int]:
    """
    After patch embedding (Conv2d(C, d_model, kernel=patch_size, stride=patch_size)):
    spatial (H/P_h, W/P_w), channels d_model. Flatten to (B, d_model, num_patches).
    With [CLS] we have (B, L, d_model) where L = num_patches + 1.
    Return (B, L, d_model) — the shape that goes into the transformer encoder.
    """
    # TODO: L = vit2d_sequence_length(H, W, patch_size); return (B, L, d_model)
    raise NotImplementedError


def vit2d_patch_embed_num_params(
    C: int, d_model: int, patch_size: int | Tuple[int, int], bias: bool = True
) -> int:
    """
    Conv2d for patch embedding: in_channels=C, out_channels=d_model, kernel=patch_size, stride=patch_size.
    Params: C * d_model * P_h * P_w + (d_model if bias).
    """
    # TODO: return parameter count
    raise NotImplementedError


# ---------------------------------------------------------------------------
# 3D ViT (video): input (B, C, T, H, W)
# ---------------------------------------------------------------------------


def vit3d_num_patches(
    T: int, H: int, W: int,
    patch_size: int | Tuple[int, int, int],
) -> int:
    """
    Number of patches (excluding [CLS]) for 3D video.
    patch_size can be int (P x P x P) or (P_t, P_h, P_w).
    """
    # TODO: return (T/P_t) * (H/P_h) * (W/P_w) as integer
    raise NotImplementedError


def vit3d_sequence_length(
    T: int, H: int, W: int,
    patch_size: int | Tuple[int, int, int],
) -> int:
    """Sequence length: num_patches + 1."""
    # TODO: return vit3d_num_patches(...) + 1
    raise NotImplementedError


def vit3d_patch_embed_output_shape(
    B: int, C: int, T: int, H: int, W: int,
    patch_size: int | Tuple[int, int, int],
    d_model: int,
) -> Tuple[int, int, int]:
    """
    After 3D patch embedding: (B, d_model, T/P_t, H/P_h, W/P_w) → flatten → (B, L-1, d_model); add [CLS] → (B, L, d_model).
    Return (B, L, d_model).
    """
    # TODO: L = vit3d_sequence_length(T, H, W, patch_size); return (B, L, d_model)
    raise NotImplementedError


def vit3d_patch_embed_num_params(
    C: int, d_model: int, patch_size: int | Tuple[int, int, int], bias: bool = True
) -> int:
    """
    Conv3d for 3D patch embedding: C → d_model, kernel=patch_size, stride=patch_size.
    Params: C * d_model * P_t * P_h * P_w + (d_model if bias).
    """
    # TODO: return parameter count
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Full ViT param count (simplified: embed + N encoder blocks + MLP head)
# ---------------------------------------------------------------------------


def vit2d_encoder_params(
    num_layers: int,
    d_model: int,
    num_heads: int,
    d_ff: int,
    patch_embed_params: int,
) -> int:
    """
    Total encoder params: patch_embed + num_layers * (MHA + FFN + 2*LayerNorm) + final LayerNorm.
    Simplified: patch_embed + num_layers * encoder_block_params + 2*d_model (final LN).
    You can use the same encoder_block formula from 02_transformer_computation (MHA + FFN + 4*d_model).
    """
    # TODO: from 02: mha = 4*d_model^2+4*d_model, ffn = 768*3072+3072+3072*768+768 style; LN = 4*d_model per block.
    # So per block: 4*d_model^2+4*d_model + d_model*d_ff+d_ff+d_ff*d_model+d_model + 4*d_model
    # Return patch_embed_params + num_layers * (that) + 2*d_model
    raise NotImplementedError


if __name__ == "__main__":
    # 2D: 224x224, patch 16 → 14*14 = 196 patches, L = 197
    assert vit2d_num_patches(224, 224, 16) == 14 * 14
    assert vit2d_sequence_length(224, 224, 16) == 14 * 14 + 1
    assert vit2d_patch_embed_output_shape(2, 3, 224, 224, 16, 768) == (2, 197, 768)
    assert vit2d_patch_embed_num_params(3, 768, 16, bias=True) == 3 * 768 * 16 * 16 + 768

    # 3D: 8 frames 224x224, patch (2, 16, 16) → 4*14*14 = 784 patches, L = 785
    assert vit3d_num_patches(8, 224, 224, (2, 16, 16)) == 4 * 14 * 14
    assert vit3d_sequence_length(8, 224, 224, (2, 16, 16)) == 4 * 14 * 14 + 1
    assert vit3d_patch_embed_output_shape(2, 3, 8, 224, 224, (2, 16, 16), 768) == (2, 785, 768)
    assert vit3d_patch_embed_num_params(3, 768, (2, 16, 16), bias=True) == 3 * 768 * 2 * 16 * 16 + 768

    # Encoder params (sanity): per block = MHA + FFN + 2*LayerNorm (same as 02)
    per_block = (
        4 * 768 * 768 + 4 * 768
        + 768 * 3072 + 3072 + 3072 * 768 + 768
        + 4 * 768
    )
    pe = vit2d_patch_embed_num_params(3, 768, 16, True)
    assert vit2d_encoder_params(12, 768, 12, 3072, pe) == pe + 12 * per_block + 2 * 768

    print("03_vision_transformer_computation OK.")
