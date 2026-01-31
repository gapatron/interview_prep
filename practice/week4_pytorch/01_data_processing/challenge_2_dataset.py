"""
01 — Custom Dataset & DataLoader
================================
Learning goal: Implement the standard PyTorch pattern for batched data: a Dataset
that returns (sample, label) per index, and a DataLoader that batches and shuffles.

Implement:
  - TensorDataset(X, y): __len__ → len(X); __getitem__(idx) → (X[idx], y[idx]).
  - create_sample_loader(batch_size, num_samples): create X (num_samples, 10), y (num_samples,) with class indices 0 or 1;
    wrap in TensorDataset and return DataLoader(..., batch_size=batch_size, shuffle=True).

Labels for classification should be integer indices (e.g. long); CrossEntropyLoss expects that.
Run the block at the bottom to verify.
"""

import torch
from torch.utils.data import Dataset, DataLoader


class TensorDataset(Dataset):
    """Dataset that returns (X[idx], y[idx]) for each index."""

    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        assert len(X) == len(y), "X and y must have same length"
        self.X = X
        self.y = y

    def __len__(self):
        pass

    def __getitem__(self, idx: int):
        pass


def create_sample_loader(batch_size: int = 4, num_samples: int = 20):
    """Create synthetic (X, y), wrap in TensorDataset, return DataLoader with given batch_size and shuffle=True."""
    pass


if __name__ == "__main__":
    ds = TensorDataset(torch.randn(10, 5), torch.randint(0, 2, (10,)))
    assert len(ds) == 10, "TensorDataset __len__ must match number of samples"
    x, y = ds[0]
    assert x.shape == (5,) and y.ndim == 0, "__getitem__ must return (sample, label)"
    assert y.dtype in (torch.int64, torch.long), "Labels must be long for CrossEntropyLoss"

    loader = create_sample_loader(batch_size=4, num_samples=20)
    batches = list(loader)
    assert len(batches) >= 1, "create_sample_loader must return a non-empty DataLoader"
    bx, by = batches[0]
    assert bx.shape[0] == by.shape[0], "Batch size of X and y must match"
    assert by.dtype in (torch.int64, torch.long), "Labels must be long"
    assert by.min() >= 0 and by.max() < 2, "Binary labels must be 0 or 1"
    print("01 — Dataset & DataLoader: OK.")
