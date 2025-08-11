import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
import sys
from pathlib import Path
from config_utils import load_cfg, ensure_dir

# =======================
# 1. Dataset定義
# =======================
class LooksDataset(Dataset):
    def __init__(self, csv_file, image_root, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_root = image_root
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_root, self.data.iloc[idx]['file'])
        image = Image.open(img_path).convert('RGB')
        label = int(self.data.iloc[idx]['looks'])

        if self.transform:
            image = self.transform(image)

        return image, label

# =======================
# 2. Transform定義
# =======================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# =======================
# 3. Hyperparameter & path
# =======================
root, cfg = load_cfg()
paths, tr = cfg["paths"], cfg["train"]

log_file = (root / paths["logs_dir"] / "resnet18_log.txt")
ensure_dir(log_file.parent)

sys.stdout = open(log_file, "w", buffering=1, encoding="utf-8")

train_csv  = root / paths["train_csv"]
test_csv   = root / paths["test_csv"]
image_root = root / paths["image_root"]


BATCH_SIZE  = tr["batch_size"]
NUM_EPOCHS  = tr["num_epochs"]
LR          = tr["learning_rate"]
NUM_WORKERS = tr["num_workers"]

# =======================
# 4. DataLoader
# =======================
train_dataset = LooksDataset(train_csv, image_root, transform)
test_dataset = LooksDataset(test_csv, image_root, transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# =======================
# 5. Model定義
# =======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

# =======================
# 6. Loss & 最適化
# =======================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# =======================
# 7. 学習Loop
# =======================
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    acc = 100 * correct / total
    print(f"[Epoch {epoch+1}] Loss: {total_loss:.4f} | Train Accuracy: {acc:.2f}%")

    # Validation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = 100 * correct / total
    print(f"Validation Accuracy: {val_acc:.2f}%\n")

# =======================
# 8. Model保存
# =======================
model_out = root / paths["models_create_dir"] / "resnet18_looks_classifier.pth"
ensure_dir(model_out.parent)
torch.save(model.state_dict(), model_out)
print(f"saved model: {model_out}")
sys.stdout.close()