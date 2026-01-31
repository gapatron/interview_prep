"""Tests for 10_diffusion_flow solutions."""

import sys
from pathlib import Path

import torch

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "10_diffusion_flow"))


def test_noise_prediction_shape():
    from solution_noise_prediction import NoisePredictionNet
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NoisePredictionNet(3, 32, 64).to(device)
    x_t = torch.randn(2, 3, 8, 8, device=device)
    t = torch.randint(0, 1000, (2,), device=device)
    pred = model(x_t, t)
    assert pred.shape == x_t.shape, f"Expected {x_t.shape}, got {pred.shape}"


def test_ddpm_train_step_loss_scalar():
    from solution_ddpm_step import ddpm_train_step, get_alpha_bars
    from solution_noise_prediction import NoisePredictionNet
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    T = 100
    alpha_bar = get_alpha_bars(T, device=device)
    model = NoisePredictionNet(3, 32, 64).to(device)
    x0 = torch.randn(4, 3, 8, 8, device=device)
    t = torch.randint(0, T, (4,), device=device)
    loss = ddpm_train_step(model, x0, t, alpha_bar, device)
    assert loss.dim() == 0 and loss.item() >= 0 and not torch.isnan(loss).any()


def test_diffusion_bug_1_alpha_shape():
    from solution_diffusion_bug_1 import make_x_t, get_alpha_bars
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    alpha_bar = get_alpha_bars(100, device=device)
    x0 = torch.randn(4, 3, 8, 8, device=device)
    t = torch.randint(0, 100, (4,), device=device)
    eps = torch.randn_like(x0, device=device)
    x_t = make_x_t(x0, t, alpha_bar, eps, device)
    assert x_t.shape == x0.shape
