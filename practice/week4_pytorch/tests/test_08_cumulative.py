"""Tests for 08_cumulative solutions."""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "08_cumulative"))


def test_end_to_end_main():
    from solution_end_to_end import main
    main()


def test_two_stage_returns_acc_and_freeze():
    from solution_two_stage import two_stage_train, BackboneHead
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BackboneHead(8, 16, 2).to(device)
    train_loader = DataLoader(TensorDataset(torch.randn(64, 8), torch.randint(0, 2, (64,))), batch_size=8, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.randn(32, 8), torch.randint(0, 2, (32,))), batch_size=8)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    final_acc = two_stage_train(model, train_loader, val_loader, criterion, device, stage1_epochs=1, stage2_epochs=1)
    assert 0 <= final_acc <= 1
    assert not any(p.requires_grad for p in model.backbone.parameters()), "Backbone should be frozen after stage 2"
