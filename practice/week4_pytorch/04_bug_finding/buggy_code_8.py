"""
Bug 8 — Target shape for CrossEntropyLoss
-----------------------------------------
Concept: CrossEntropyLoss expects targets shape (N,) and dtype long, not (N, 1) or float. Loss must be a scalar for backward.
What goes wrong: Targets are created with shape (N, 1) or wrong dtype; fix by using (N,) and long. Run, find, fix. Compare with solution_bug_8.py.
"""

import torch
import torch.nn as nn


if __name__ == "__main__":
    model = nn.Linear(10, 3)
    logits = model(torch.randn(4, 10))
    # BUG: targets shape (4, 1) — CrossEntropy expects (4,) and long
    targets = torch.randint(0, 3, (4, 1))
    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits, targets)
    loss.backward()
    print("OK.")
