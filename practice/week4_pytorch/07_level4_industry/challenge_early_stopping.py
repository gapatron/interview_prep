"""
Level 4 — Early Stopping
=========================
Stop training when validation metric does not improve for `patience` epochs.
  - After each epoch: if val_acc > best_acc then best_acc = val_acc, reset counter; else counter += 1
  - If counter >= patience: break and return best model / best_acc
Implement early_stopping_loop: train for up to max_epochs, validate each epoch,
track best_val_acc and no_improve_count; break when no_improve_count >= patience.
Return (best_val_acc, epoch_stopped).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total += loss.item()
        n += 1
    return total / max(n, 1)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total_loss += criterion(logits, y).item()
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
    return total_loss / max(len(loader), 1), correct / max(total, 1)


def early_stopping_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    max_epochs: int,
    patience: int,
) -> tuple[float, int]:
    """
    Train with early stopping. Return (best_val_acc, epoch_stopped).
    If val accuracy improves: reset no_improve_count. Else: no_improve_count += 1.
    Stop when no_improve_count >= patience.
    """
    # TODO: best_acc = 0.0, no_improve_count = 0
    # TODO: for epoch in range(max_epochs): train_one_epoch(...); val_loss, val_acc = validate(...)
    # TODO:   if val_acc > best_acc: best_acc = val_acc; no_improve_count = 0
    # TODO:   else: no_improve_count += 1
    # TODO:   if no_improve_count >= patience: return best_acc, epoch
    # TODO: return best_acc, max_epochs - 1
    pass


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 2)).to(device)
    train_ds = TensorDataset(torch.randn(64, 8), torch.randint(0, 2, (64,)))
    val_ds = TensorDataset(torch.randn(32, 8), torch.randint(0, 2, (32,)))
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    best_acc, stopped_epoch = early_stopping_loop(
        model, train_loader, val_loader, criterion, optimizer, device, max_epochs=10, patience=2
    )
    assert 0 <= best_acc <= 1 and stopped_epoch < 10
    print("Early stopping OK.")
