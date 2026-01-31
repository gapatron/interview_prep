"""
01 — Tensors & data processing
==============================
Learning goal: Device selection, 4D batch shapes, concat vs stack, and broadcasting
a conditioning vector onto a feature map so you can concatenate along channels.

Implement:
  - get_device() → cuda if available else cpu.
  - create_batch_tensor(batch_size, channels, height, width) → 4D zeros (N, C, H, W).
  - concat_along_channels(a, b): a (N,C1,H,W), b (N,C2,H,W) → (N, C1+C2, H, W) along channel dim.
  - condition_to_feature_map(x, cond): x (N,C,H,W), cond (N,D) → broadcast cond to (N,D,H,W), concat with x → (N,C+D,H,W).
  - stack_batches(list_of_3D_tensors): each (C,H,W) → stack to (B,C,H,W).

Run the block at the bottom to check shapes.
"""

import torch


def get_device():
    """Return the device to use (cuda if available, else cpu)."""
    pass


def create_batch_tensor(batch_size: int, channels: int, height: int, width: int):
    """Create a 4D tensor of shape (batch_size, channels, height, width)."""
    pass


def concat_along_channels(a: torch.Tensor, b: torch.Tensor):
    """Concatenate two 4D tensors along the channel dimension. a (N,C1,H,W), b (N,C2,H,W) → (N,C1+C2,H,W)."""
    pass


def condition_to_feature_map(x: torch.Tensor, cond: torch.Tensor):
    """Broadcast cond (N,D) to (N,D,H,W) and concatenate with x (N,C,H,W) along channels → (N,C+D,H,W)."""
    pass


def stack_batches(list_of_tensors):
    """Stack a list of 3D tensors (C,H,W) into a single 4D tensor (B,C,H,W)."""
    pass


if __name__ == "__main__":
    device = get_device()
    assert device is not None, "get_device() must return a device"
    assert device.type in ("cuda", "cpu"), f"device.type must be cuda or cpu, got {device.type}"

    t = create_batch_tensor(4, 3, 8, 8)
    assert t.shape == (4, 3, 8, 8), f"Expected (4,3,8,8), got {t.shape}"
    assert t.dtype == torch.float32, f"create_batch_tensor should return float32, got {t.dtype}"
    assert t.dim() == 4, "create_batch_tensor must return 4D tensor"

    a, b = torch.randn(2, 3, 4, 4), torch.randn(2, 5, 4, 4)
    c = concat_along_channels(a, b)
    assert c.shape == (2, 8, 4, 4), f"Expected (2,8,4,4), got {c.shape}"
    assert c.size(1) == a.size(1) + b.size(1), "concat_along_channels: channel dim must be C1+C2"

    x, cond = torch.randn(2, 3, 4, 4), torch.randn(2, 6)
    out = condition_to_feature_map(x, cond)
    assert out.shape == (2, 9, 4, 4), f"Expected (2,9,4,4), got {out.shape}"
    assert out.size(1) == x.size(1) + cond.size(1), "condition_to_feature_map: channels must be C+D"

    tensors = [torch.randn(3, 4, 4) for _ in range(5)]
    stacked = stack_batches(tensors)
    assert stacked.shape == (5, 3, 4, 4), f"Expected (5,3,4,4), got {stacked.shape}"
    assert stacked.dim() == 4, "stack_batches must return 4D (B,C,H,W)"

    print("01 — Tensors & data processing: OK.")
