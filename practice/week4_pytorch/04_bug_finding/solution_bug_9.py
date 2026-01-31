"""
Solution: Bug 9 — model.eval() in validation.
Fix: Call model.eval() at the start of the validation function (before the loop).
Why: Same as Bug 5: BN and Dropout must use eval behavior for consistent validation metrics.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def validate(model, loader, device):
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / max(total, 1)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.Sequential(
        nn.Linear(8, 16),
        nn.BatchNorm1d(16),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(16, 2),
    ).to(device)
    ds = TensorDataset(torch.randn(32, 8), torch.randint(0, 2, (32,)))
    loader = DataLoader(ds, batch_size=8)
    acc = validate(model, loader, device)
    print(f"Val acc: {acc:.4f}")
