"""
Solution: Device, 4D tensors, concat along channels, broadcast cond to feature map, stack batches.
Takeaway: cat preserves dims and concatenates; stack adds a new dim. Condition: unsqueeze to (N,D,1,1), expand to (N,D,H,W).
"""

import torch


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_batch_tensor(batch_size: int, channels: int, height: int, width: int):
    return torch.zeros(batch_size, channels, height, width)


def concat_along_channels(a: torch.Tensor, b: torch.Tensor):
    return torch.cat([a, b], dim=1)


def condition_to_feature_map(x: torch.Tensor, cond: torch.Tensor):
    N, C, H, W = x.shape
    D = cond.shape[1]
    cond = cond.unsqueeze(2).unsqueeze(3).expand(N, D, H, W)
    return torch.cat([x, cond], dim=1)


def stack_batches(list_of_tensors):
    return torch.stack(list_of_tensors, dim=0)


if __name__ == "__main__":
    device = get_device()
    assert device.type in ("cuda", "cpu")
    t = create_batch_tensor(4, 3, 8, 8)
    assert t.shape == (4, 3, 8, 8) and t.dtype == torch.float32 and not torch.isnan(t).any()
    a, b = torch.randn(2, 3, 4, 4), torch.randn(2, 5, 4, 4)
    c = concat_along_channels(a, b)
    assert c.shape == (2, 8, 4, 4) and c.size(1) == a.size(1) + b.size(1)
    x, cond = torch.randn(2, 3, 4, 4), torch.randn(2, 6)
    out = condition_to_feature_map(x, cond)
    assert out.shape == (2, 9, 4, 4) and out.size(1) == x.size(1) + cond.size(1)
    tensors = [torch.randn(3, 4, 4) for _ in range(5)]
    stacked = stack_batches(tensors)
    assert stacked.shape == (5, 3, 4, 4) and stacked.dim() == 4
    print("01 — Tensors & data processing: OK.")
