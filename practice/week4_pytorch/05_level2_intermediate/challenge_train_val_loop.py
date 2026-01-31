"""
05 — Full train + validation loop
==================================
Learning goal: Separate training (gradients, model.train()) from validation (no grad, model.eval());
compute accuracy from argmax(logits, dim=1) == targets; drive a multi-epoch loop from one function.

Implement:
  - validate(model, loader, criterion, device) → (avg_loss, accuracy). Use model.eval(), torch.no_grad(), and count correct predictions.
  - train_for_epochs(...): for each epoch call train_one_epoch then validate; print train_loss, val_loss, val_acc.

train_one_epoch is already provided; reuse the same pattern from 03_training. Run the block at the bottom to verify.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    """Train one epoch. Return average loss."""
    model.train()
    total_loss, n = 0.0, 0
    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Validation: no grad, model.eval(). Return (avg_loss, accuracy)."""
    # TODO: model.eval(), torch.no_grad(), loop over loader
    # TODO: compute loss and count correct predictions (argmax(logits, dim=1) == batch_y)
    # TODO: return total_loss / n_batches, correct / total
    pass


def train_for_epochs(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epochs: int,
):
    """For each epoch: train_one_epoch, validate, print train_loss val_loss val_acc."""
    # TODO: for epoch in range(epochs): train_one_epoch(...), validate(...), print
    pass


if __name__ == "__main__":
    from torch.utils.data import TensorDataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 2)).to(device)
    train_ds = TensorDataset(torch.randn(64, 10), torch.randint(0, 2, (64,)))
    val_ds = TensorDataset(torch.randn(32, 10), torch.randint(0, 2, (32,)))
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    train_for_epochs(model, train_loader, val_loader, criterion, optimizer, device, 3)
    print("Done.")
