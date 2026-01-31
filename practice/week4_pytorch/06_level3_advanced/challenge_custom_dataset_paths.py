"""
Level 3 — Custom Dataset from Paths (Industry Pattern)
======================================================
Implement a Dataset that takes a list of (path, label) and returns (sample, label)
in __getitem__. For a runnable challenge without real files, use in-memory
"paths" that map to pre-loaded tensors (dict path -> tensor). This pattern
appears in image loaders: path -> load image -> transform -> return.
  - __init__(self, path_label_pairs, path_to_tensor_dict)
  - __len__ -> len(path_label_pairs)
  - __getitem__(self, idx) -> (tensor, label); tensor = path_to_tensor_dict[path]
Implement PathLabelDataset and run the assert.
"""

import torch
from torch.utils.data import Dataset


class PathLabelDataset(Dataset):
    """
    path_label_pairs: list of (path_key, label)
    path_to_tensor: dict mapping path_key -> tensor (e.g. preloaded "images")
    """

    def __init__(self, path_label_pairs: list, path_to_tensor: dict):
        # TODO: store path_label_pairs and path_to_tensor
        pass

    def __len__(self) -> int:
        # TODO: return len(self.path_label_pairs)
        pass

    def __getitem__(self, idx: int):
        # TODO: path, label = self.path_label_pairs[idx]
        # TODO: x = self.path_to_tensor[path]  # copy or reference OK for demo
        # TODO: return x, label
        pass


if __name__ == "__main__":
    path_to_tensor = {"a": torch.tensor([1.0, 2.0]), "b": torch.tensor([3.0, 4.0]), "c": torch.tensor([5.0, 6.0])}
    pairs = [("a", 0), ("b", 1), ("c", 0)]
    ds = PathLabelDataset(pairs, path_to_tensor)
    assert len(ds) == 3
    x, y = ds[1]
    assert x.tolist() == [3.0, 4.0] and y == 1
    print("PathLabelDataset OK.")
