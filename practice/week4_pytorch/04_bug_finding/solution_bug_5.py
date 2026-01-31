"""
Solution: Bug 5 — Forgetting model.eval().
Fix: Call model.eval() at the start of the validation loop (or function).
Why: BatchNorm uses running stats and Dropout is disabled only in eval mode.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.Sequential(nn.Linear(10, 20), nn.Dropout(0.5), nn.ReLU(), nn.Linear(20, 2)).to(device)
    loader = DataLoader(TensorDataset(torch.randn(32, 10), torch.randint(0, 2, (32,))), batch_size=8)
    val_loss, val_acc = validate(model, loader, nn.CrossEntropyLoss(), device)
    print(f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
