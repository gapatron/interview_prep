"""Solution: End-to-end pipeline (data → model → train → val → save best)."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from pathlib import Path


def set_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X = torch.randn(200, 10)
    y = torch.randint(0, 2, (200,))
    dataset = TensorDataset(X, y)
    train_ds, val_ds = random_split(dataset, [160, 40])
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16)

    model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 2)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    best_acc = 0.0
    save_path = Path("/tmp/best.pt")
    for epoch in range(5):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}/5  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_acc": val_acc}, save_path)
    print(f"Best val_acc: {best_acc:.4f}")


if __name__ == "__main__":
    main()
