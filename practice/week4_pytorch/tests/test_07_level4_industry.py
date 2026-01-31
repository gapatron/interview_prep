"""Tests for 07_level4_industry solutions."""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "07_level4_industry"))


def test_early_stopping_returns_valid():
    from solution_early_stopping import early_stopping_loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 2)).to(device)
    train_loader = DataLoader(TensorDataset(torch.randn(64, 8), torch.randint(0, 2, (64,))), batch_size=8, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.randn(32, 8), torch.randint(0, 2, (32,))), batch_size=8)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    best_acc, stopped_epoch = early_stopping_loop(model, train_loader, val_loader, criterion, optimizer, device, max_epochs=5, patience=2)
    assert 0 <= best_acc <= 1
    assert 0 <= stopped_epoch < 5


def test_lr_scheduler_final_lr():
    from solution_lr_scheduler import train_with_scheduler
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.Linear(4, 2).to(device)
    loader = DataLoader(TensorDataset(torch.randn(24, 4), torch.randint(0, 2, (24,))), batch_size=8)
    criterion = nn.CrossEntropyLoss()
    final_lr = train_with_scheduler(model, loader, criterion, device, epochs=3, lr=0.1, step_size=1, gamma=0.5)
    assert abs(final_lr - 0.0125) < 1e-5, f"Expected 0.0125, got {final_lr}"


def test_full_pipeline_runs():
    from solution_full_pipeline import run_full_pipeline
    run_full_pipeline(epochs=2, save_dir="/tmp/pytorch_pipeline_test")
    p = Path("/tmp/pytorch_pipeline_test/best.pt")
    assert p.exists()
    state = torch.load(p, map_location="cpu", weights_only=False)
    assert "model" in state and "val_acc" in state
