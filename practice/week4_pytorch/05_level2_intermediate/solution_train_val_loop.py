"""Solution: Full train + validation loop."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, n = 0.0, 0
    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == batch_y).sum().item()
            total += batch_y.size(0)
    n = len(loader)
    return total_loss / max(n, 1), correct / max(total, 1)


def train_for_epochs(model, train_loader, val_loader, criterion, optimizer, device, epochs):
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}/{epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")


if __name__ == "__main__":
    from torch.utils.data import TensorDataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 2)).to(device)
    train_loader = DataLoader(TensorDataset(torch.randn(64, 10), torch.randint(0, 2, (64,))), batch_size=8, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.randn(32, 10), torch.randint(0, 2, (32,))), batch_size=8)
    train_for_epochs(model, train_loader, val_loader, nn.CrossEntropyLoss(), optim.Adam(model.parameters(), lr=1e-3), device, 3)
    print("Done.")
