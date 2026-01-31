"""
Cumulative — Two-Stage Training (Freeze Backbone, Finetune Head)
================================================================
Industry pattern: train full model for a few epochs, then freeze backbone
and train only the head (e.g. for transfer learning or stability).
  1. Build Backbone + Head model (from Level 2).
  2. Stage 1: train full model for stage1_epochs (all params).
  3. Stage 2: freeze backbone (for p in backbone.parameters(): p.requires_grad = False),
     train only head for stage2_epochs.
  4. Return final val accuracy after stage 2.
Implement two_stage_train and run the assert. Use train_one_epoch and validate from Level 2.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class Backbone(nn.Module):
    def __init__(self, in_dim, hidden):
        super().__init__()
        self.fc = nn.Linear(in_dim, hidden)

    def forward(self, x):
        return torch.relu(self.fc(x))


class Head(nn.Module):
    def __init__(self, hidden, num_classes):
        super().__init__()
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, z):
        return self.fc(z)


class BackboneHead(nn.Module):
    def __init__(self, in_dim, hidden, num_classes):
        super().__init__()
        self.backbone = Backbone(in_dim, hidden)
        self.head = Head(hidden, num_classes)

    def forward(self, x):
        return self.head(self.backbone(x))


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total += loss.item()
        n += 1
    return total / max(n, 1)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total_loss += criterion(logits, y).item()
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
    return total_loss / max(len(loader), 1), correct / max(total, 1)


def two_stage_train(
    model: BackboneHead,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    stage1_epochs: int = 2,
    stage2_epochs: int = 2,
    lr: float = 1e-3,
) -> float:
    """
    Stage 1: train full model for stage1_epochs.
    Stage 2: freeze backbone, train only head for stage2_epochs.
    Return final val accuracy.
    """
    # TODO: Stage 1 — optimizer on all params, train stage1_epochs
    # TODO: Stage 2 — for p in model.backbone.parameters(): p.requires_grad = False
    # TODO: optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    # TODO: train stage2_epochs, then validate and return val_acc
    pass


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BackboneHead(8, 16, 2).to(device)
    train_ds = TensorDataset(torch.randn(64, 8), torch.randint(0, 2, (64,)))
    val_ds = TensorDataset(torch.randn(32, 8), torch.randint(0, 2, (32,)))
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8)
    criterion = nn.CrossEntropyLoss()
    final_acc = two_stage_train(model, train_loader, val_loader, criterion, device, stage1_epochs=2, stage2_epochs=2)
    assert 0 <= final_acc <= 1
    # Backbone should be frozen
    assert not any(p.requires_grad for p in model.backbone.parameters())
    print("Two-stage train OK.")
