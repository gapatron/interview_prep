"""
Level 4 — LR Scheduler in the Training Loop
============================================
Use StepLR (or CosineAnnealingLR): step once per epoch after optimizer.step().
Wire scheduler into a small training loop: for each epoch train_one_epoch(...),
then scheduler.step(). Return final learning rate (from optimizer.param_groups[0]['lr']).
Implement train_with_scheduler(epochs=3, step_size=1, gamma=0.5, lr=0.1).
Assert that after 3 epochs the LR has been reduced three times (e.g. 0.1 -> 0.05 -> 0.025 -> 0.0125).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
    return 0.0


def train_with_scheduler(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epochs: int = 3,
    lr: float = 0.1,
    step_size: int = 1,
    gamma: float = 0.5,
) -> float:
    """Train for epochs, stepping scheduler once per epoch. Return final LR."""
    # TODO: optimizer = optim.SGD(model.parameters(), lr=lr)
    # TODO: scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    # TODO: for epoch in range(epochs): train_one_epoch(...); scheduler.step()
    # TODO: return optimizer.param_groups[0]["lr"]
    pass


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.Linear(4, 2).to(device)
    ds = TensorDataset(torch.randn(24, 4), torch.randint(0, 2, (24,)))
    loader = DataLoader(ds, batch_size=8)
    criterion = nn.CrossEntropyLoss()
    final_lr = train_with_scheduler(model, loader, criterion, device, epochs=3, lr=0.1, step_size=1, gamma=0.5)
    assert abs(final_lr - 0.0125) < 1e-6
    print("LR scheduler OK.")
