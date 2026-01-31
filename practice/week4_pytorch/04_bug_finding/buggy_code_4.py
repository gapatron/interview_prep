"""
Bug 4 — Device mismatch
----------------------
Concept: Model and every batch tensor must be on the same device (e.g. .to(device)). Otherwise forward or loss will raise.
What goes wrong: Model is moved to device but batches are not; you get a device mismatch error. Run, find, fix. Compare with solution_bug_4.py.
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
        # BUG: x and y are still on CPU
        logits = model(x)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("Done.")


if __name__ == "__main__":
    main()
