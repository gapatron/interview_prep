"""
06 — Gradient clipping
======================
Learning goal: Clip gradient norms before optimizer.step() to stabilize training (common in RNNs and deep nets). Order: backward → clip_grad_norm_ → step.

Implement: train_one_epoch_with_clip(..., max_norm=1.0). Same loop as train_one_epoch, but after loss.backward() and before optimizer.step() call nn.utils.clip_grad_norm_(model.parameters(), max_norm). Return average loss.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


def train_one_epoch_with_clip(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    max_norm: float = 1.0,
) -> float:
    """Train one epoch with gradient clipping. Return average loss."""
    model.train()
    total_loss, n = 0.0, 0
    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        # TODO: clip gradients: torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


if __name__ == "__main__":
    from torch.utils.data import TensorDataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 2)).to(device)
    loader = DataLoader(TensorDataset(torch.randn(32, 10), torch.randint(0, 2, (32,))), batch_size=8, shuffle=True)
    loss = train_one_epoch_with_clip(model, loader, nn.CrossEntropyLoss(), optim.Adam(model.parameters(), lr=1e-3), device, max_norm=1.0)
    print(f"Epoch loss: {loss:.4f}")
