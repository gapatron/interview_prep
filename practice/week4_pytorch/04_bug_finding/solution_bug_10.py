"""
Solution: Bug 10 — Scheduler step.
Fix: Call scheduler.step() once per epoch, in the outer loop after train_one_epoch (or after the batch loop), not inside the batch loop.
Why: StepLR and similar schedulers are designed to step once per epoch; stepping every batch changes the schedule incorrectly.
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


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.Linear(4, 2).to(device)
    opt = optim.SGD(model.parameters(), lr=0.1)
    sched = optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.5)
    ds = TensorDataset(torch.randn(16, 4), torch.randint(0, 2, (16,)))
    loader = DataLoader(ds, batch_size=4)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(3):
        train_one_epoch(model, loader, criterion, opt, device)
        sched.step()
    print("OK.")
