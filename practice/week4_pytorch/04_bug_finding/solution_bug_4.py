"""
Solution: Bug 4 — Device mismatch.
Fix: Move each batch to device: x, y = x.to(device), y.to(device) before forward.
Why: Model and all inputs must be on the same device.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.Linear(10, 2).to(device)
    loader = DataLoader(TensorDataset(torch.randn(16, 10), torch.randint(0, 2, (16,))), batch_size=4)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("Done.")


if __name__ == "__main__":
    main()
