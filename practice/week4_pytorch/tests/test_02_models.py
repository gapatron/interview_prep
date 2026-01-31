"""Tests for 02_models solutions. Validates shapes and gradient flow."""

import sys
from pathlib import Path

import pytest
import torch

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "02_models"))


def test_concat_block_shape():
    from solution_1_concatenation import ConcatBlock
    block = ConcatBlock(10, 5, 8)
    x_a = torch.randn(4, 10)
    x_b = torch.randn(4, 5)
    out = block(x_a, x_b)
    assert out.shape == (4, 8), f"Expected (4, 8), got {out.shape}"


def test_concat_block_gradient_flow():
    from solution_1_concatenation import ConcatBlock
    block = ConcatBlock(10, 5, 8)
    x_a = torch.randn(2, 10, requires_grad=True)
    x_b = torch.randn(2, 5, requires_grad=True)
    out = block(x_a, x_b)
    out.sum().backward()
    assert x_a.grad is not None and x_b.grad is not None


def test_conditional_mlp_shape():
    from solution_2_conditioning import ConditionalMLP
    model = ConditionalMLP(20, 4, 32, 3)
    x = torch.randn(5, 20)
    c = torch.randn(5, 4)
    out = model(x, c)
    assert out.shape == (5, 3), f"Expected (5, 3), got {out.shape}"


def test_conditional_mlp_no_nan():
    from solution_2_conditioning import ConditionalMLP
    model = ConditionalMLP(20, 4, 32, 3)
    x = torch.randn(2, 20)
    c = torch.randn(2, 4)
    out = model(x, c)
    assert not torch.isnan(out).any() and not torch.isinf(out).any()


def test_full_model_shape():
    from solution_3_connecting import FullModel
    model = FullModel(10, 20, 2)
    x = torch.randn(4, 10)
    out = model(x)
    assert out.shape == (4, 2), f"Expected (4, 2), got {out.shape}"


def test_full_model_encoder_decoder_connected():
    from solution_3_connecting import FullModel, Encoder, Decoder
    model = FullModel(10, 20, 2)
    x = torch.randn(2, 10, requires_grad=True)
    out = model(x)
    out.sum().backward()
    assert x.grad is not None
