"""
Solution: Bug 1 — Labels dtype.
Fix: Use integer class indices, e.g. torch.randint(0, 2, (32,)) so dtype is long.
Why: CrossEntropyLoss expects class indices (long), not float or one-hot.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class SmallClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 2)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallClassifier().to(device)
    X = torch.randn(32, 10)
    y = torch.randint(0, 2, (32,))
    loader = DataLoader(TensorDataset(X, y), batch_size=4, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()

    print("Done.")


if __name__ == "__main__":
    main()
