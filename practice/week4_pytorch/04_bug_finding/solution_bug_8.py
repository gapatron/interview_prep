"""
Solution: Bug 8 — Target shape.
Fix: Use targets with shape (N,) and dtype long, e.g. torch.randint(0, C, (N,), dtype=torch.long).
Why: CrossEntropyLoss expects class indices (N,) long; (N,1) or float causes shape/type errors.
"""

import torch
import torch.nn as nn


if __name__ == "__main__":
    model = nn.Linear(10, 3)
    logits = model(torch.randn(4, 10))
    targets = torch.randint(0, 3, (4,), dtype=torch.long)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits, targets)
    loss.backward()
    assert loss.dim() == 0
    print("OK.")
