"""
Solution: Mixed precision (FP16) training.
Takeaway: Use autocast() for forward+loss; GradScaler for scale(loss).backward(), step(optimizer), update(). Skip when not CUDA.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def train_one_epoch_amp(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss, n = 0.0, 0
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        if device.type == "cuda" and scaler is not None:
            with torch.amp.autocast("cuda"):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        else:
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        n += 1

    return total_loss / max(n, 1)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2)).to(device)
    ds = TensorDataset(torch.randn(16, 4), torch.randint(0, 2, (16,)))
    loader = DataLoader(ds, batch_size=4)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss = train_one_epoch_amp(model, loader, criterion, optimizer, device)
    assert isinstance(loss, float) and loss >= 0
    print("Mixed precision OK.")
