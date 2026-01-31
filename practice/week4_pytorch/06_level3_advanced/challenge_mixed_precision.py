"""
06 — Mixed precision training (FP16)
====================================
Learning goal: Use FP16 (half precision) in the forward pass to save memory and speed up
on GPUs that support it. GradScaler scales the loss so small gradients don’t underflow;
autocast runs the forward in FP16 where safe.

Implement:
  - train_one_epoch_amp(model, loader, criterion, optimizer, device) → average loss.
  - Create a GradScaler (e.g. torch.amp.GradScaler("cuda")). In the batch loop: with torch.amp.autocast("cuda"): forward and loss;
    then scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update(); optimizer.zero_grad() (or zero_grad before backward).
  - Only use autocast and GradScaler when device.type == "cuda"; otherwise run normal FP32.

Run the assert at the bottom. See PyTorch_Mastery_Guide.md “Mixed precision” and “Quizzes.”
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def train_one_epoch_amp(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    """Train one epoch with mixed precision (FP16) when device is CUDA. Return average loss."""
    model.train()
    total_loss, n = 0.0, 0
    # TODO: if device.type == "cuda": create GradScaler; use autocast() for forward+loss, scaler.scale(loss).backward(), scaler.step(optimizer), scaler.update()
    # TODO: else: normal FP32 loop (forward, loss, backward, step)
    # TODO: optimizer.zero_grad() at the start of each batch (or before backward)
    pass


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2)).to(device)
    ds = TensorDataset(torch.randn(16, 4), torch.randint(0, 2, (16,)))
    loader = DataLoader(ds, batch_size=4)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss = train_one_epoch_amp(model, loader, criterion, optimizer, device)
    assert isinstance(loss, float) and loss >= 0
    print("Mixed precision OK.")
