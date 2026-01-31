"""Tests for 11_transformers solutions."""

import sys
from pathlib import Path

import torch

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "11_transformers"))


def test_scaled_dot_product_attention_shape():
    from solution_scaled_dot_product_attention import scaled_dot_product_attention
    B, L, d_k = 2, 4, 8
    Q = torch.randn(B, L, d_k)
    K = torch.randn(B, L, d_k)
    V = torch.randn(B, L, d_k)
    out, attn = scaled_dot_product_attention(Q, K, V)
    assert out.shape == (B, L, d_k)
    assert attn.shape == (B, L, L)
    assert torch.allclose(attn.sum(dim=-1), torch.ones(B, L))


def test_multihead_attention_shape():
    from solution_multihead_attention import MultiHeadAttention
    B, L, d_model, H = 2, 4, 8, 2
    mha = MultiHeadAttention(d_model, H)
    x = torch.randn(B, L, d_model)
    out = mha(x)
    assert out.shape == (B, L, d_model)


def test_encoder_block_shape():
    from solution_encoder_block import TransformerEncoderBlock
    B, L, d_model, H, d_ff = 2, 4, 8, 2, 32
    block = TransformerEncoderBlock(d_model, H, d_ff)
    x = torch.randn(B, L, d_model)
    out = block(x)
    assert out.shape == (B, L, d_model)
