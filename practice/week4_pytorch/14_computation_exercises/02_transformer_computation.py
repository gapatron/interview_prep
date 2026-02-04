"""
14 — Transformer computation (1D sequences)
============================================
Learning goal: Shapes at every step, parameter counts for Multi-Head Attention (MHA)
and Feed-Forward Network (FFN), and total params per encoder block. Sequence format (B, L, d_model).

Steps to master:
  1. Embedding: (B, L) indices → (B, L, d_model)
  2. Q, K, V projections: (B, L, d_model) → each (B, L, d_model); then reshape to (B, H, L, head_dim)
  3. Attention scores: Q @ K^T → (B, H, L, L)
  4. Attention output: softmax(scores) @ V → (B, H, L, head_dim) → concat → (B, L, d_model)
  5. Output projection W_out: (B, L, d_model) → (B, L, d_model)
  6. FFN: Linear(d_model → d_ff) → GELU → Linear(d_ff → d_model)

Implement the functions below. All dimensions are explicit (B, L, d_model, H, d_ff).
See: Computation_Exercises_Guide.md (Transformers section).
"""

from typing import Tuple


def embedding_shape(B: int, L: int, d_model: int) -> Tuple[int, int, int]:
    """Output shape of token embedding: (B, L) → (B, L, d_model)."""
    # TODO: return (B, L, d_model)
    raise NotImplementedError


def mha_qkv_shapes_after_projection(
    B: int, L: int, d_model: int
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int], Tuple[int, int, int]]:
    """After W_q, W_k, W_v (each d_model → d_model): Q, K, V each (B, L, d_model). Return ((B,L,d_model), (B,L,d_model), (B,L,d_model))."""
    # TODO: return shapes of Q, K, V after linear projection (before head split)
    raise NotImplementedError


def mha_attention_scores_shape(B: int, H: int, L: int) -> Tuple[int, int, int, int]:
    """Q @ K^T per head: (B, H, L, head_dim) @ (B, H, head_dim, L) → (B, H, L, L)."""
    # TODO: return (B, H, L, L)
    raise NotImplementedError


def mha_output_shape(B: int, L: int, d_model: int) -> Tuple[int, int, int]:
    """After concat heads and W_out: (B, L, d_model)."""
    # TODO: return (B, L, d_model)
    raise NotImplementedError


def mha_num_params(d_model: int, num_heads: int, bias: bool = True) -> int:
    """
    Multi-Head Attention parameter count.
    W_q, W_k, W_v: each d_model * d_model; W_out: d_model * d_model.
    Total: 4 * d_model^2 + (4 * d_model if bias). (We assume d_model % num_heads == 0; head_dim not in param count.)
    """
    # TODO: return total parameters for MHA (4 linear layers: Q, K, V, out)
    raise NotImplementedError


def ffn_num_params(d_model: int, d_ff: int, bias: bool = True) -> int:
    """
    Feed-Forward Network: Linear(d_model → d_ff) + Linear(d_ff → d_model).
    Params: d_model*d_ff + d_ff + d_ff*d_model + d_model if bias.
    """
    # TODO: return total parameters for the two linear layers
    raise NotImplementedError


def encoder_block_num_params(
    d_model: int, num_heads: int, d_ff: int, bias: bool = True
) -> int:
    """
    One encoder block: MHA + 2 LayerNorms + FFN.
    LayerNorm params: 2*d_model per norm (weight + bias). So 2 * 2*d_model = 4*d_model for the two norms.
    Return: mha_num_params(...) + ffn_num_params(...) + 4*d_model
    """
    # TODO: MHA + FFN + two LayerNorms (each 2*d_model)
    raise NotImplementedError


def attention_flops(B: int, L: int, d_model: int) -> int:
    """
    FLOPs (mult-adds) for single-head attention (Q,K,V already projected to d_model).
    Q @ K^T: (B, L, d_model) @ (B, d_model, L) = B * L * L * d_model.
    attn @ V: (B, L, L) @ (B, L, d_model) = B * L * L * d_model.
    Total: 2 * B * L^2 * d_model. (Scaling and softmax are O(B*L^2) adds, often omitted in FLOP count.)
    """
    # TODO: return 2 * B * L * L * d_model
    raise NotImplementedError


if __name__ == "__main__":
    B, L, d_model, H, d_ff = 2, 128, 768, 12, 3072

    assert embedding_shape(B, L, d_model) == (2, 128, 768)
    assert mha_qkv_shapes_after_projection(B, L, d_model) == ((2, 128, 768), (2, 128, 768), (2, 128, 768))
    assert mha_attention_scores_shape(B, H, L) == (2, 12, 128, 128)
    assert mha_output_shape(B, L, d_model) == (2, 128, 768)

    # MHA: 4 * (d_model * d_model) + 4 * d_model = 4 * 768^2 + 4*768 = 2,359,296 + 3,072 = 2,362,368
    assert mha_num_params(768, 12, bias=True) == 4 * 768 * 768 + 4 * 768
    # FFN: 768*3072 + 3072 + 3072*768 + 768 = 2*768*3072 + 3072 + 768
    assert ffn_num_params(768, 3072, bias=True) == 768 * 3072 + 3072 + 3072 * 768 + 768
    assert encoder_block_num_params(768, 12, 3072, bias=True) == (
        mha_num_params(768, 12, True) + ffn_num_params(768, 3072, True) + 4 * 768
    )
    assert attention_flops(B, L, d_model) == 2 * B * L * L * d_model

    print("02_transformer_computation OK.")
