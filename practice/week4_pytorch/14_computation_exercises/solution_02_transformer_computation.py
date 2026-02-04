"""Solution: Transformer shapes and parameter counts (1D sequence)."""

from typing import Tuple


def embedding_shape(B: int, L: int, d_model: int) -> Tuple[int, int, int]:
    return (B, L, d_model)


def mha_qkv_shapes_after_projection(
    B: int, L: int, d_model: int
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int], Tuple[int, int, int]]:
    q = (B, L, d_model)
    k = (B, L, d_model)
    v = (B, L, d_model)
    return (q, k, v)


def mha_attention_scores_shape(B: int, H: int, L: int) -> Tuple[int, int, int, int]:
    return (B, H, L, L)


def mha_output_shape(B: int, L: int, d_model: int) -> Tuple[int, int, int]:
    return (B, L, d_model)


def mha_num_params(d_model: int, num_heads: int, bias: bool = True) -> int:
    # W_q, W_k, W_v, W_out: each d_model x d_model; each has d_model bias
    return 4 * (d_model * d_model) + (4 * d_model if bias else 0)


def ffn_num_params(d_model: int, d_ff: int, bias: bool = True) -> int:
    # Linear1: d_model * d_ff + d_ff, Linear2: d_ff * d_model + d_model
    return d_model * d_ff + (d_ff if bias else 0) + d_ff * d_model + (d_model if bias else 0)


def encoder_block_num_params(
    d_model: int, num_heads: int, d_ff: int, bias: bool = True
) -> int:
    ln_params = 2 * (2 * d_model)  # two LayerNorms, each weight and bias of size d_model
    return mha_num_params(d_model, num_heads, bias) + ffn_num_params(d_model, d_ff, bias) + ln_params


def attention_flops(B: int, L: int, d_model: int) -> int:
    return 2 * B * L * L * d_model
