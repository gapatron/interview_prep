"""
Solution: TensorDataset and create_sample_loader.
Takeaway: Dataset needs __len__ and __getitem__; DataLoader handles batching and shuffle. Use integer labels for classification.
"""

import torch
from torch.utils.data import Dataset, DataLoader


class TensorDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        assert len(X) == len(y)
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def create_sample_loader(batch_size: int = 4, num_samples: int = 20):
    X = torch.randn(num_samples, 10)
    y = torch.randint(0, 2, (num_samples,))
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


if __name__ == "__main__":
    ds = TensorDataset(torch.randn(10, 5), torch.randint(0, 2, (10,)))
    assert len(ds) == 10
    x, y = ds[0]
    assert x.shape == (5,) and y.ndim == 0
    loader = create_sample_loader(batch_size=4, num_samples=20)
    for bx, by in loader:
        assert bx.shape[0] == by.shape[0]
        break
    print("01 — Dataset & DataLoader: OK.")
