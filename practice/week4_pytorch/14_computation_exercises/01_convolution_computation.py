"""
14 — Convolution computation (1D, 2D, 3D)
==========================================
Learning goal: Output shape formula, parameter count, and FLOPs for Conv1d, Conv2d, Conv3d
with explicit dimensions. Master these for interview questions.

Formulas (no dilation for simplicity; dilation multiplies effective kernel size):
  - Output length (each dimension): out = floor((in + 2*pad - kernel) / stride) + 1
  - Parameters (no bias): C_in * C_out * K1 * K2 * ...  (+ C_out if bias=True)
  - FLOPs (per position, then total): each output element = C_in * (K1*K2*...) * C_out mult-adds

Implement the functions below. Use integer arithmetic (no torch needed for the formula functions).
See: Computation_Exercises_Guide.md (Convolutions section).
"""

import math
from typing import Tuple


# ---------------------------------------------------------------------------
# 1D Convolution (e.g. audio, time series): input (B, C_in, L)
# ---------------------------------------------------------------------------


def conv1d_output_length(L_in: int, kernel_size: int, stride: int = 1, padding: int = 0) -> int:
    """
    Output length for Conv1d.
    L_in: input length (single dimension).
    Returns: L_out (integer).
    Formula: L_out = floor((L_in + 2*padding - kernel_size) / stride) + 1
    """
    # TODO: return the integer L_out
    raise NotImplementedError


def conv1d_num_params(C_in: int, C_out: int, kernel_size: int, bias: bool = True) -> int:
    """
    Number of learnable parameters for Conv1d.
    Weights: C_in * C_out * kernel_size. Bias: C_out if bias=True.
    """
    # TODO: return total parameter count
    raise NotImplementedError


def conv1d_flops(C_in: int, C_out: int, kernel_size: int, L_out: int) -> int:
    """
    FLOPs (mult-adds) for Conv1d forward pass.
    Per output position: C_in * kernel_size * C_out. Total: multiply by L_out.
    """
    # TODO: return total FLOPs
    raise NotImplementedError


# ---------------------------------------------------------------------------
# 2D Convolution (e.g. images): input (B, C_in, H, W)
# ---------------------------------------------------------------------------


def conv2d_output_shape(
    H_in: int,
    W_in: int,
    kernel_size: int | Tuple[int, int],
    stride: int | Tuple[int, int] = 1,
    padding: int | Tuple[int, int] = 0,
) -> Tuple[int, int]:
    """
    Output (H_out, W_out) for Conv2d.
    kernel_size, stride, padding can be int (same for H and W) or (int, int).
    """
    # TODO: normalize to (kH, kW), (sH, sW), (pH, pW) then apply formula per dimension
    raise NotImplementedError


def conv2d_num_params(
    C_in: int,
    C_out: int,
    kernel_size: int | Tuple[int, int],
    bias: bool = True,
) -> int:
    """Parameters: C_in * C_out * kH * kW + (C_out if bias)."""
    # TODO: return total parameter count
    raise NotImplementedError


def conv2d_flops(
    C_in: int,
    C_out: int,
    kernel_size: int | Tuple[int, int],
    H_out: int,
    W_out: int,
) -> int:
    """FLOPs for Conv2d: (C_in * kH * kW * C_out) * H_out * W_out."""
    # TODO: return total FLOPs
    raise NotImplementedError


# ---------------------------------------------------------------------------
# 3D Convolution (e.g. video): input (B, C_in, T, H, W)
# ---------------------------------------------------------------------------


def conv3d_output_shape(
    T_in: int,
    H_in: int,
    W_in: int,
    kernel_size: int | Tuple[int, int, int],
    stride: int | Tuple[int, int, int] = 1,
    padding: int | Tuple[int, int, int] = 0,
) -> Tuple[int, int, int]:
    """
    Output (T_out, H_out, W_out) for Conv3d.
    Typical: kernel_size=(3,3,3) or (kT, kH, kW); same formula per dimension.
    """
    # TODO: return (T_out, H_out, W_out)
    raise NotImplementedError


def conv3d_num_params(
    C_in: int,
    C_out: int,
    kernel_size: int | Tuple[int, int, int],
    bias: bool = True,
) -> int:
    """Parameters: C_in * C_out * kT * kH * kW + (C_out if bias)."""
    # TODO: return total parameter count
    raise NotImplementedError


def conv3d_flops(
    C_in: int,
    C_out: int,
    kernel_size: int | Tuple[int, int, int],
    T_out: int,
    H_out: int,
    W_out: int,
) -> int:
    """FLOPs for Conv3d: (C_in * kT * kH * kW * C_out) * T_out * H_out * W_out."""
    # TODO: return total FLOPs
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Sanity check with PyTorch (optional but recommended)
# ---------------------------------------------------------------------------


def _check_conv1d():
    import torch
    import torch.nn as nn
    B, C_in, C_out, L = 2, 4, 8, 100
    k, s, p = 5, 2, 1
    conv = nn.Conv1d(C_in, C_out, k, stride=s, padding=p)
    x = torch.randn(B, C_in, L)
    y = conv(x)
    L_out = y.size(-1)
    assert L_out == conv1d_output_length(L, k, s, p), (L_out, conv1d_output_length(L, k, s, p))
    nparams = sum(p.numel() for p in conv.parameters())
    assert nparams == conv1d_num_params(C_in, C_out, k, bias=True), (nparams, conv1d_num_params(C_in, C_out, k, True))
    assert conv1d_flops(C_in, C_out, k, L_out) == C_in * k * C_out * L_out


def _check_conv2d():
    import torch
    import torch.nn as nn
    B, C_in, C_out, H, W = 2, 3, 64, 32, 32
    k, s, p = 3, 1, 1
    conv = nn.Conv2d(C_in, C_out, k, stride=s, padding=p)
    x = torch.randn(B, C_in, H, W)
    y = conv(x)
    Ho, Wo = y.size(2), y.size(3)
    assert (Ho, Wo) == conv2d_output_shape(H, W, k, s, p)
    nparams = sum(p.numel() for p in conv.parameters())
    assert nparams == conv2d_num_params(C_in, C_out, k, bias=True)
    assert conv2d_flops(C_in, C_out, k, Ho, Wo) == C_in * k * k * C_out * Ho * Wo


def _check_conv3d():
    import torch
    import torch.nn as nn
    B, C_in, C_out, T, H, W = 2, 3, 16, 8, 32, 32
    k = (3, 3, 3)
    conv = nn.Conv3d(C_in, C_out, k, stride=1, padding=1)
    x = torch.randn(B, C_in, T, H, W)
    y = conv(x)
    To, Ho, Wo = y.size(2), y.size(3), y.size(4)
    assert (To, Ho, Wo) == conv3d_output_shape(T, H, W, k, 1, 1)
    nparams = sum(p.numel() for p in conv.parameters())
    assert nparams == conv3d_num_params(C_in, C_out, k, bias=True)


if __name__ == "__main__":
    # 1D explicit
    assert conv1d_output_length(100, 5, 2, 1) == (100 + 2 * 1 - 5) // 2 + 1
    assert conv1d_num_params(4, 8, 5, bias=True) == 4 * 8 * 5 + 8
    assert conv1d_flops(4, 8, 5, 50) == 4 * 5 * 8 * 50

    # 2D explicit: 32x32, 3x3, stride 1, padding 1 -> 32x32
    assert conv2d_output_shape(32, 32, 3, 1, 1) == (32, 32)
    assert conv2d_num_params(3, 64, 3, bias=True) == 3 * 64 * 9 + 64
    assert conv2d_flops(3, 64, 3, 32, 32) == 3 * 9 * 64 * 32 * 32

    # 3D explicit: 8x32x32, kernel 3x3x3, stride 1, padding 1 -> 8x32x32
    assert conv3d_output_shape(8, 32, 32, (3, 3, 3), 1, 1) == (8, 32, 32)
    assert conv3d_num_params(3, 16, (3, 3, 3), bias=True) == 3 * 16 * 27 + 16
    assert conv3d_flops(3, 16, (3, 3, 3), 8, 32, 32) == 3 * 27 * 16 * 8 * 32 * 32

    _check_conv1d()
    _check_conv2d()
    _check_conv3d()
    print("01_convolution_computation OK.")
