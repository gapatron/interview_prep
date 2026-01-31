"""Tests for 03_training solution. Validates training step and loss."""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "03_training"))


def test_train_one_epoch_returns_float():
    from solution_training_loop import train_one_epoch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 2)).to(device)
    X = torch.randn(32, 10)
    y = torch.randint(0, 2, (32,))
    loader = DataLoader(TensorDataset(X, y), batch_size=4, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss = train_one_epoch(model, loader, criterion, optimizer, device)
    assert isinstance(loss, float), f"Expected float, got {type(loss)}"
    assert loss >= 0, f"Loss should be non-negative, got {loss}"
    assert not (loss != loss), "Loss must not be NaN"


def test_train_one_epoch_reduces_loss():
    from solution_training_loop import train_one_epoch
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 2)).to(device)
    X = torch.randn(64, 10)
    y = torch.randint(0, 2, (64,))
    loader = DataLoader(TensorDataset(X, y), batch_size=8, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    loss1 = train_one_epoch(model, loader, criterion, optimizer, device)
    loss2 = train_one_epoch(model, loader, criterion, optimizer, device)
    # Loss should generally decrease or stay similar (not necessarily strictly)
    assert loss2 <= loss1 * 1.5 + 0.01, "Training should not explode; loss2 should be in reasonable range"
