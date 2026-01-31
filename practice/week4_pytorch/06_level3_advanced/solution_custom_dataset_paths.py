"""Solution: Custom Dataset from path–label pairs and path->tensor mapping."""

import torch
from torch.utils.data import Dataset


class PathLabelDataset(Dataset):
    def __init__(self, path_label_pairs: list, path_to_tensor: dict):
        self.path_label_pairs = path_label_pairs
        self.path_to_tensor = path_to_tensor

    def __len__(self) -> int:
        return len(self.path_label_pairs)

    def __getitem__(self, idx: int):
        path, label = self.path_label_pairs[idx]
        x = self.path_to_tensor[path]
        return x.clone(), label


if __name__ == "__main__":
    path_to_tensor = {"a": torch.tensor([1.0, 2.0]), "b": torch.tensor([3.0, 4.0]), "c": torch.tensor([5.0, 6.0])}
    pairs = [("a", 0), ("b", 1), ("c", 0)]
    ds = PathLabelDataset(pairs, path_to_tensor)
    assert len(ds) == 3
    x, y = ds[1]
    assert x.tolist() == [3.0, 4.0] and y == 1
    print("PathLabelDataset OK.")
