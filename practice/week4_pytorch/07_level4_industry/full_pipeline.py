"""
07 — Full industry-style pipeline
==================================
Learning goal: Tie seed, device, data, model, train+val loop, and best-model saving into one runnable script. No new concepts—wire Level 1–3 pieces together.

Implement (fill TODOs):
  1. set_seed; device; build_dataloaders (TensorDataset, random_split, DataLoader train/val).
  2. Model, criterion, optimizer.
  3. Loop: for each epoch run train_one_epoch, validate; if val_acc > best_acc then save checkpoint (model.state_dict(), epoch, val_acc) and update best_acc.
  4. Print best_val_acc.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from pathlib import Path


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_dataloaders(num_samples=200, batch_size=16, val_ratio=0.2):
    """Build train and val DataLoaders. TODO: create TensorDataset, split, DataLoader each."""
    # TODO: X = torch.randn(num_samples, 10), y = torch.randint(0, 2, (num_samples,))
    # TODO: dataset = TensorDataset(X, y); n_val = int(num_samples * val_ratio); n_train = num_samples - n_val
    # TODO: train_ds, val_ds = random_split(dataset, [n_train, n_val])
    # TODO: train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    # TODO: val_loader = DataLoader(val_ds, batch_size=batch_size)
    # TODO: return train_loader, val_loader
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


def run_full_pipeline(epochs=5, save_dir="/tmp/pytorch_pipeline"):
    """Full pipeline: seed, device, data, model, train+val, save best."""
    set_seed(42)
    device = get_device()
    # TODO: train_loader, val_loader = build_dataloaders()
    # TODO: model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 2)).to(device)
    # TODO: criterion = nn.CrossEntropyLoss(); optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # TODO: best_acc = 0.0; Path(save_dir).mkdir(parents=True, exist_ok=True)
    # TODO: for epoch in range(epochs): train_one_epoch(...); val_loss, val_acc = validate(...); print(...)
    # TODO: if val_acc > best_acc: best_acc = val_acc; torch.save({"model": model.state_dict(), "epoch": epoch, "val_acc": val_acc}, Path(save_dir) / "best.pt")
    # TODO: print(f"Best val_acc: {best_acc:.4f}")
    pass


if __name__ == "__main__":
    run_full_pipeline(epochs=3)
