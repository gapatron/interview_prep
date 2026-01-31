"""
Bug 9 — Forgetting model.eval() (BatchNorm / Dropout)
-----------------------------------------------------
Concept: Same as Bug 5: validation must use model.eval() so BN and Dropout behave correctly.
What goes wrong: The validation function never calls model.eval() before the loop. Run, find, fix. Compare with solution_bug_9.py.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def validate(model, loader, device):
    # BUG: we never call model.eval(), so BN uses batch stats, Dropout is on
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
