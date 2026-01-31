"""
Bug 10 — Scheduler step placement
---------------------------------
Concept: StepLR (and most schedulers) should be stepped once per epoch, after the training loop for that epoch, not every batch.
What goes wrong: scheduler.step() is called inside the batch loop; the learning rate then changes every batch instead of every epoch. Run, find, fix. Compare with solution_bug_10.py.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def train_one_epoch(model, loader, criterion, optimizer, scheduler, device):
    model.train()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        # BUG: scheduler stepped every batch; should be once per epoch
        scheduler.step()
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
        train_one_epoch(model, loader, criterion, opt, sched, device)
        # Intended: lr should change once per epoch (0.1 -> 0.05 -> 0.025)
    print("OK.")
