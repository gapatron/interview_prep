"""Tests for 09_expert solutions."""

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "09_expert"))


def test_custom_autograd_matches_leaky_relu():
    from solution_custom_autograd import leaky_relu_custom
    x = torch.randn(2, 4, requires_grad=True)
    y = leaky_relu_custom(x, neg_slope=0.1)
    y.sum().backward()
    assert x.grad is not None
    x2 = x.detach().clone().requires_grad_(True)
    y2 = torch.nn.functional.leaky_relu(x2, 0.1)
    y2.sum().backward()
    torch.testing.assert_close(x.grad, x2.grad)


def test_distributed_ready_model_on_device():
    from solution_distributed_ready import setup_model_for_distributed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.Linear(4, 2)
    model = setup_model_for_distributed(model, device)
    assert next(model.parameters()).device.type == device.type
