"""
03 — Training loop (one epoch)
=============================
Learning goal: Implement the core training step: for each batch, zero gradients,
forward pass, loss, backward, optimizer step; move data to device. Return average loss.

The loop is already written; your job is to ensure the correct order and that
batch tensors are on the same device as the model (otherwise you get device mismatch errors).
If anything is missing or in the wrong order, fix it.

Run the block at the bottom to train one epoch and print the loss.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch. Return average loss. Use model.train(); move batches to device; zero_grad → forward → loss → backward → step."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


if __name__ == "__main__":
    from torch.utils.data import TensorDataset

    class SmallMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 20)
            self.fc2 = nn.Linear(20, 2)

        def forward(self, x):
            return self.fc2(torch.relu(self.fc1(x)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallMLP().to(device)
    X = torch.randn(32, 10)
    y = torch.randint(0, 2, (32,))
    loader = DataLoader(TensorDataset(X, y), batch_size=4, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss = train_one_epoch(model, loader, criterion, optimizer, device)
    assert isinstance(loss, float), "train_one_epoch must return a float"
    assert loss >= 0, f"Loss must be non-negative, got {loss}"
    assert not (loss != loss), "Loss must not be NaN"
    print(f"03 — One epoch loss: {loss:.4f}")
