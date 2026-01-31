"""Solution: Training with gradient clipping."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


def train_one_epoch_with_clip(model, loader, criterion, optimizer, device, max_norm=1.0):
    model.train()
    total_loss, n = 0.0, 0
    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)
