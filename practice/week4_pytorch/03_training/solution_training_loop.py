"""
Solution: One-epoch training loop.
Takeaway: Order is zero_grad → forward → loss → backward → step; move batch to device before forward.
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
    loader = DataLoader(TensorDataset(torch.randn(32, 10), torch.randint(0, 2, (32,))), batch_size=4, shuffle=True)
    loss = train_one_epoch(model, loader, nn.CrossEntropyLoss(), optim.Adam(model.parameters(), lr=1e-3), device)
    assert isinstance(loss, float) and loss >= 0
    print(f"03 — One epoch loss: {loss:.4f}")
