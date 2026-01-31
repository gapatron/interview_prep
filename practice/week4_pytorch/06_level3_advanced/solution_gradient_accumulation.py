"""Solution: Gradient accumulation — zero_grad every accum_steps, step every accum_steps, scale loss."""

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
    model.train()
    total_loss, n = 0.0, 0
    for batch_idx, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        if batch_idx % accum_steps == 0:
            optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y) / accum_steps
        loss.backward()
        if (batch_idx + 1) % accum_steps == 0:
            optimizer.step()
        total_loss += loss.item() * accum_steps
        n += 1
    if n % accum_steps != 0:
        optimizer.step()
    return total_loss / max(n, 1)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2)).to(device)
    ds = TensorDataset(torch.randn(16, 4), torch.randint(0, 2, (16,)))
    loader = DataLoader(ds, batch_size=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss = train_one_epoch_accum(model, loader, criterion, optimizer, device, accum_steps=2)
    assert isinstance(loss, float), "Must return float"
    assert loss >= 0 and not (loss != loss), "Loss must be non-negative and not NaN"
    print("Gradient accumulation OK.")
