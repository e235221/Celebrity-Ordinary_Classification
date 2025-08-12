from pathlib import Path
import torch
import torch.nn as nn

def setup_logging(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    import sys
    sys.stdout = open(log_path, "w", buffering=1, encoding="utf-8")

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total if total else 0.0

def train(model, train_loader, val_loader, device, epochs=10, lr=1e-4, optimizer_cls=torch.optim.Adam):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optimizer_cls(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_acc = 100.0 * correct / total if total else 0.0
        val_acc = evaluate(model, val_loader, device)
        print(f"[Epoch {epoch+1}] Loss: {total_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
    return model

def save_model(model, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f"saved model: {out_path}")