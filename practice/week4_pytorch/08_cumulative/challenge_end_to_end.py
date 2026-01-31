"""
08 — End-to-end pipeline (cumulative)
=====================================
Learning goal: No new concepts. Implement a single script that: creates data (TensorDataset, random_split, DataLoader train/val), builds model and optimizer, runs train_one_epoch and validate each epoch, and saves the best checkpoint when val_acc improves.

Implement (fill TODOs):
  1. Device, seed; X, y; TensorDataset; random_split; DataLoader train/val.
  2. Model (e.g. nn.Sequential(Linear, ReLU, Linear)), criterion, optimizer.
  3. best_acc = 0; for each epoch: train_one_epoch, validate; if val_acc > best_acc: save checkpoint, update best_acc.
  4. Print best_val_acc.

Reuse logic from 01_data_processing, 05_level2_intermediate (train_one_epoch, validate), and 07_level4_industry.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from pathlib import Path


def set_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    # TODO: X = torch.randn(200, 10), y = torch.randint(0, 2, (200,))
    # TODO: dataset = TensorDataset(X, y); train_ds, val_ds = random_split(dataset, [160, 40])
    # TODO: train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    # TODO: val_loader = DataLoader(val_ds, batch_size=16)
    pass

    # Model
    # TODO: model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 2)).to(device)
    pass

    # Optimizer, criterion
    # TODO: criterion = nn.CrossEntropyLoss(); optimizer = optim.Adam(model.parameters(), lr=1e-3)
    pass

    # Train + validate + save best
    # TODO: best_acc = 0.0; save_path = Path("/tmp/best.pt")
    # TODO: for epoch in range(5): train_one_epoch(...); val_loss, val_acc = validate(...); if val_acc > best_acc: best_acc = val_acc; torch.save({"model": model.state_dict(), "epoch": epoch, "val_acc": val_acc}, save_path)
    # TODO: print(f"Best val_acc: {best_acc:.4f}")
    pass


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


if __name__ == "__main__":
    main()
