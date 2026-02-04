"""Solution: Complexity formulas and interview Q&A."""

from typing import Tuple


def attention_time_complexity(L: int, d_model: int) -> str:
    return "O(L^2 * d_model)"


def attention_space_complexity(B: int, H: int, L: int) -> str:
    return "O(B * H * L^2)"


def conv2d_time_complexity_per_layer(
    C_in: int, C_out: int, K: int, H_out: int, W_out: int
) -> str:
    return "O(C_in * C_out * K^2 * H * W)"


def why_attention_is_expensive(L: int) -> str:
    return "Attention computes pairwise scores over L positions, giving an L×L matrix, so time and space are O(L^2)."


def conv_vs_attention_compute(
    L_conv: int,
    C: int,
    K: int,
    L_seq: int,
    d: int,
) -> Tuple[str, str]:
    return ("C*C*K*L_conv", "2*L_seq^2*d")
