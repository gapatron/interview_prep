"""
06 — Gradient accumulation (effective larger batch)
====================================================
Learning goal: Simulate a larger batch by accumulating gradients over several mini-batches
before calling optimizer.step(). Saves memory (smaller batch size) while matching the effective batch size. Scale loss by 1/accum_steps so gradient magnitude is correct.

Implement:
  - train_one_epoch_accum(..., accum_steps): zero_grad only before the first batch of each group (e.g. when batch_idx % accum_steps == 0); loss = criterion(...) / accum_steps; backward(); step only after every accum_steps batches (e.g. when (batch_idx + 1) % accum_steps == 0). For reporting, accumulate total_loss as loss.item() * accum_steps. Handle the last incomplete group (step once more if needed).
Run the assert. See PyTorch_Mastery_Guide.md “Gradient accumulation” and “Quizzes.”
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def train_one_epoch_accum(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    accum_steps: int = 4,
) -> float:
    """Train one epoch with gradient accumulation every accum_steps batches."""
    model.train()
    total_loss, n = 0.0, 0
    # TODO: for batch_idx, (x, y) in enumerate(loader):
    # TODO:   x, y = x.to(device), y.to(device)
    # TODO:   if batch_idx % accum_steps == 0: optimizer.zero_grad()
    # TODO:   logits = model(x); loss = criterion(logits, y) / accum_steps; loss.backward()
    # TODO:   if (batch_idx + 1) % accum_steps == 0: optimizer.step()
    # TODO:   total_loss += loss.item() * accum_steps; n += 1
    # TODO: return total_loss / max(n, 1)
    pass


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2)).to(device)
    ds = TensorDataset(torch.randn(16, 4), torch.randint(0, 2, (16,)))
    loader = DataLoader(ds, batch_size=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss = train_one_epoch_accum(model, loader, criterion, optimizer, device, accum_steps=2)
    assert isinstance(loss, float), "train_one_epoch_accum must return float"
    assert loss >= 0 and not (loss != loss), "Loss must be non-negative and not NaN"
    print("Gradient accumulation OK.")
