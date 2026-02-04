"""Solution: Convolution output shape, parameter count, and FLOPs (1D, 2D, 3D)."""

import math
from typing import Tuple


def _to_pair(x: int | Tuple[int, int]) -> Tuple[int, int]:
    if isinstance(x, int):
        return (x, x)
    return x


def _to_triple(x: int | Tuple[int, int, int]) -> Tuple[int, int, int]:
    if isinstance(x, int):
        return (x, x, x)
    return x


def conv1d_output_length(L_in: int, kernel_size: int, stride: int = 1, padding: int = 0) -> int:
    return (L_in + 2 * padding - kernel_size) // stride + 1


def conv1d_num_params(C_in: int, C_out: int, kernel_size: int, bias: bool = True) -> int:
    w = C_in * C_out * kernel_size
    return w + (C_out if bias else 0)


def conv1d_flops(C_in: int, C_out: int, kernel_size: int, L_out: int) -> int:
    return C_in * kernel_size * C_out * L_out


def conv2d_output_shape(
    H_in: int,
    W_in: int,
    kernel_size: int | Tuple[int, int],
    stride: int | Tuple[int, int] = 1,
    padding: int | Tuple[int, int] = 0,
) -> Tuple[int, int]:
    kH, kW = _to_pair(kernel_size)
    sH, sW = _to_pair(stride)
    pH, pW = _to_pair(padding)
    H_out = (H_in + 2 * pH - kH) // sH + 1
    W_out = (W_in + 2 * pW - kW) // sW + 1
    return (H_out, W_out)


def conv2d_num_params(
    C_in: int,
    C_out: int,
    kernel_size: int | Tuple[int, int],
    bias: bool = True,
) -> int:
    kH, kW = _to_pair(kernel_size)
    w = C_in * C_out * kH * kW
    return w + (C_out if bias else 0)


def conv2d_flops(
    C_in: int,
    C_out: int,
    kernel_size: int | Tuple[int, int],
    H_out: int,
    W_out: int,
) -> int:
    kH, kW = _to_pair(kernel_size)
    return C_in * kH * kW * C_out * H_out * W_out


def conv3d_output_shape(
    T_in: int,
    H_in: int,
    W_in: int,
    kernel_size: int | Tuple[int, int, int],
    stride: int | Tuple[int, int, int] = 1,
    padding: int | Tuple[int, int, int] = 0,
) -> Tuple[int, int, int]:
    kT, kH, kW = _to_triple(kernel_size)
    sT, sH, sW = _to_triple(stride)
    pT, pH, pW = _to_triple(padding)
    T_out = (T_in + 2 * pT - kT) // sT + 1
    H_out = (H_in + 2 * pH - kH) // sH + 1
    W_out = (W_in + 2 * pW - kW) // sW + 1
    return (T_out, H_out, W_out)


def conv3d_num_params(
    C_in: int,
    C_out: int,
    kernel_size: int | Tuple[int, int, int],
    bias: bool = True,
) -> int:
    kT, kH, kW = _to_triple(kernel_size)
    w = C_in * C_out * kT * kH * kW
    return w + (C_out if bias else 0)


def conv3d_flops(
    C_in: int,
    C_out: int,
    kernel_size: int | Tuple[int, int, int],
    T_out: int,
    H_out: int,
    W_out: int,
) -> int:
    kT, kH, kW = _to_triple(kernel_size)
    return C_in * kT * kH * kW * C_out * T_out * H_out * W_out
