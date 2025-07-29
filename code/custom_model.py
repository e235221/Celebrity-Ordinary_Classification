import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
import pandas as pd
from PIL import Image
from torchvision.models import ResNet34_Weights
import sys

# # log path
log_file = '/home/student/e21/e215706/dm/sorce/log/custom_log.txt'
os.makedirs(os.path.dirname(log_file), exist_ok=True)
sys.stdout = open(log_file, 'w')

# -----------------------------
# 1. Dataset定義
# -----------------------------
class LooksDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.image_dir, row['file'])
        label = int(row['looks'])

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, label

# -----------------------------
# 2. Transform定義
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# -----------------------------
# 3. Dataset path
# -----------------------------
train_csv = '/home/student/e21/e215706/dm/sorce/image/csv/all_train.csv'
train_dir = '/home/student/e21/e215706/dm/sorce/image/all_image'

val_csv = '/home/student/e21/e215706/dm/sorce/image/csv/all_test.csv'
val_dir = '/home/student/e21/e215706/dm/sorce/image/all_image'

train_dataset = LooksDataset(train_csv, train_dir, transform=transform)
val_dataset = LooksDataset(val_csv, val_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

# -----------------------------
# 4. Model読み込み及び修正
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet34(weights=None)
model.fc = nn.Linear(model.fc.in_features, 18)  # 重み調整
model = model.to(device)

checkpoint_path = '/home/student/e21/e215706/dm/sorce/model/res34_fair_align_multi_7_20190809.pt'
state_dict = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(state_dict)

model.fc = nn.Linear(model.fc.in_features, 2)  # looks 0/1
model = model.to(device)

# -----------------------------
# 5. Loss & 最適化
# -----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# -----------------------------
# 6. 学習Loop
# -----------------------------
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_acc = 100. * correct / total
    print(f"[Epoch {epoch+1}] Loss: {running_loss:.4f} | Train Accuracy: {train_acc:.2f}%")

    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_acc = 100. * correct / total
    print(f"Validation Accuracy: {val_acc:.2f}%\n")

# -----------------------------
# 7. Model保存
# -----------------------------
torch.save(model.state_dict(), '/home/student/e21/e215706/dm/sorce/model/custom/custom.pth')
print("saved model custom.pth")

sys.stdout.close()
