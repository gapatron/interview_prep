"""
Edge-case, numerical sanity, device consistency, and parametrized tests.
Cross-cutting tests: batch size 1, finite outputs, device placement, gradient flow, expected errors.
"""

import sys
from pathlib import Path

import pytest
import torch

BASE = Path(__file__).resolve().parent.parent


def _path(subdir: str):
    d = BASE / subdir
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))


# ---- Edge cases: batch size 1 ----

def test_data_processing_batch_size_one():
    """Data processing solutions should handle batch size 1."""
    _path("01_data_processing")
    from solution_1_tensors import (
        create_batch_tensor,
        concat_along_channels,
        condition_to_feature_map,
        stack_batches,
    )
    t = create_batch_tensor(1, 3, 8, 8)
    assert t.shape == (1, 3, 8, 8) and t.dtype == torch.float32
    a, b = torch.randn(1, 3, 4, 4), torch.randn(1, 5, 4, 4)
    c = concat_along_channels(a, b)
    assert c.shape == (1, 8, 4, 4)
    x, cond = torch.randn(1, 3, 4, 4), torch.randn(1, 6)
    out = condition_to_feature_map(x, cond)
    assert out.shape == (1, 9, 4, 4)
    stacked = stack_batches([torch.randn(3, 4, 4)])
    assert stacked.shape == (1, 3, 4, 4)


def test_models_batch_size_one():
    """Model solutions should handle batch size 1."""
    _path("02_models")
    from solution_1_concatenation import ConcatBlock
    from solution_2_conditioning import ConditionalMLP
    from solution_3_connecting import FullModel
    block = ConcatBlock(10, 5, 8)
    out = block(torch.randn(1, 10), torch.randn(1, 5))
    assert out.shape == (1, 8)
    mlp = ConditionalMLP(20, 4, 32, 3)
    out = mlp(torch.randn(1, 20), torch.randn(1, 4))
    assert out.shape == (1, 3)
    model = FullModel(10, 20, 2)
    out = model(torch.randn(1, 10))
    assert out.shape == (1, 2)


def test_popular_architectures_batch_size_one():
    """Popular architecture solutions should handle batch size 1."""
    _path("12_popular_architectures")
    from solution_simple_cnn import SimpleCNN
    from solution_residual_block import ResidualBlock
    model = SimpleCNN(3, 10)
    x = torch.randn(1, 3, 32, 32)
    out = model(x)
    assert out.shape == (1, 10)
    block = ResidualBlock(64, 64, stride=1)
    out = block(torch.randn(1, 64, 8, 8))
    assert out.shape == (1, 64, 8, 8)


# ---- Numerical sanity ----

def test_outputs_finite_data_processing(seed):
    """Data processing outputs should be finite (no NaN/Inf)."""
    _path("01_data_processing")
    from solution_1_tensors import concat_along_channels, condition_to_feature_map, stack_batches
    a, b = torch.randn(2, 3, 4, 4), torch.randn(2, 5, 4, 4)
    c = concat_along_channels(a, b)
    assert torch.isfinite(c).all(), "concat_along_channels produced non-finite values"
    x, cond = torch.randn(2, 3, 4, 4), torch.randn(2, 6)
    out = condition_to_feature_map(x, cond)
    assert torch.isfinite(out).all(), "condition_to_feature_map produced non-finite values"


def test_outputs_finite_models(seed):
    """Model forward passes should produce finite outputs."""
    _path("02_models")
    from solution_2_conditioning import ConditionalMLP
    from solution_3_connecting import FullModel
    mlp = ConditionalMLP(20, 4, 32, 3)
    out = mlp(torch.randn(4, 20), torch.randn(4, 4))
    assert torch.isfinite(out).all(), "ConditionalMLP produced non-finite values"
    model = FullModel(10, 20, 2)
    out = model(torch.randn(4, 10))
    assert torch.isfinite(out).all(), "FullModel produced non-finite values"


def test_ddpm_step_loss_finite(device, seed):
    """DDPM training step should return finite loss."""
    _path("10_diffusion_flow")
    from solution_ddpm_step import ddpm_train_step, get_alpha_bars
    from solution_noise_prediction import NoisePredictionNet
    T = 50
    alpha_bar = get_alpha_bars(T, device=device)
    model = NoisePredictionNet(3, 32, 64).to(device)
    x0 = torch.randn(2, 3, 8, 8, device=device)
    t = torch.randint(0, T, (2,), device=device)
    loss = ddpm_train_step(model, x0, t, alpha_bar, device)
    assert torch.isfinite(loss).all() and loss.item() >= 0


# ---- Device consistency ----

def test_solutions_respect_device(device, seed):
    """Key solutions should run on fixture device and return tensors on same device."""
    _path("02_models")
    from solution_1_concatenation import ConcatBlock
    from solution_3_connecting import FullModel
    block = ConcatBlock(10, 5, 8).to(device)
    x_a = torch.randn(2, 10, device=device)
    x_b = torch.randn(2, 5, device=device)
    out = block(x_a, x_b)
    assert out.device.type == device.type, f"ConcatBlock output on {out.device}, expected {device}"
    model = FullModel(10, 20, 2).to(device)
    x = torch.randn(2, 10, device=device)
    out = model(x)
    assert out.device.type == device.type, f"FullModel output on {out.device}, expected {device}"


def test_diffusion_on_device(device, seed):
    """Diffusion solutions should run on fixture device."""
    _path("10_diffusion_flow")
    from solution_noise_prediction import NoisePredictionNet
    from solution_ddpm_step import get_alpha_bars, ddpm_train_step
    model = NoisePredictionNet(3, 32, 64).to(device)
    x_t = torch.randn(2, 3, 8, 8, device=device)
    t = torch.randint(0, 100, (2,), device=device)
    pred = model(x_t, t)
    assert pred.device.type == device.type
    alpha_bar = get_alpha_bars(100, device=device)
    x0 = torch.randn(2, 3, 8, 8, device=device)
    tt = torch.randint(0, 100, (2,), device=device)
    loss = ddpm_train_step(model, x0, tt, alpha_bar, device)
    assert loss.device.type == device.type


# ---- Gradient flow (no NaN grads) ----

def test_backward_produces_finite_grads(seed):
    """Backward should not produce NaN or Inf gradients."""
    _path("02_models")
    from solution_1_concatenation import ConcatBlock
    from solution_3_connecting import FullModel
    block = ConcatBlock(10, 5, 8)
    x_a = torch.randn(2, 10, requires_grad=True)
    x_b = torch.randn(2, 5, requires_grad=True)
    out = block(x_a, x_b)
    out.sum().backward()
    assert x_a.grad is not None and torch.isfinite(x_a.grad).all()
    assert x_b.grad is not None and torch.isfinite(x_b.grad).all()
    model = FullModel(10, 20, 2)
    x = torch.randn(2, 10, requires_grad=True)
    out = model(x)
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_transformer_encoder_gradient_flow(seed):
    """Transformer encoder block backward should produce finite grads."""
    _path("11_transformers")
    from solution_encoder_block import TransformerEncoderBlock
    block = TransformerEncoderBlock(d_model=64, num_heads=4, d_ff=128, dropout=0.0)
    x = torch.randn(2, 5, 64, requires_grad=True)
    out = block(x)
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


# ---- Parametrized ----

@pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
def test_concat_along_channels_batch_sizes(batch_size):
    """concat_along_channels should work for various batch sizes."""
    _path("01_data_processing")
    from solution_1_tensors import concat_along_channels
    a = torch.randn(batch_size, 3, 4, 4)
    b = torch.randn(batch_size, 5, 4, 4)
    c = concat_along_channels(a, b)
    assert c.shape == (batch_size, 8, 4, 4)


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_full_model_batch_sizes(batch_size):
    """FullModel should work for batch sizes 1, 2, 4."""
    _path("02_models")
    from solution_3_connecting import FullModel
    model = FullModel(10, 20, 2)
    x = torch.randn(batch_size, 10)
    out = model(x)
    assert out.shape == (batch_size, 2)


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_simple_cnn_batch_sizes(batch_size):
    """SimpleCNN should work for batch sizes 1, 2, 4."""
    _path("12_popular_architectures")
    from solution_simple_cnn import SimpleCNN
    model = SimpleCNN(3, 10)
    x = torch.randn(batch_size, 3, 32, 32)
    out = model(x)
    assert out.shape == (batch_size, 10)


# ---- Expected errors ----

def test_concat_mismatched_batch_raises():
    """concat_along_channels with mismatched batch sizes should raise."""
    _path("01_data_processing")
    from solution_1_tensors import concat_along_channels
    a = torch.randn(2, 3, 4, 4)
    b = torch.randn(3, 5, 4, 4)  # different N
    with pytest.raises(RuntimeError):
        concat_along_channels(a, b)


def test_condition_mismatched_batch_raises():
    """condition_to_feature_map with mismatched N should raise."""
    _path("01_data_processing")
    from solution_1_tensors import condition_to_feature_map
    x = torch.randn(2, 3, 4, 4)
    cond = torch.randn(3, 6)  # different N
    with pytest.raises(RuntimeError):
        condition_to_feature_map(x, cond)


# ---- Challenge files raise NotImplementedError ----

def test_challenge_simple_cnn_raises_not_implemented():
    """Challenge SimpleCNN should raise NotImplementedError before implementation."""
    _path("12_popular_architectures")
    with pytest.raises(NotImplementedError):
        from challenge_simple_cnn import SimpleCNN
        SimpleCNN(3, 10)


def test_challenge_residual_block_raises_not_implemented():
    """Challenge ResidualBlock should raise NotImplementedError before implementation."""
    _path("12_popular_architectures")
    with pytest.raises(NotImplementedError):
        from challenge_residual_block import ResidualBlock
        ResidualBlock(64, 64, stride=1)
