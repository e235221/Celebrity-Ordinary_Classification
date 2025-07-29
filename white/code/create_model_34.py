import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import ResNet34_Weights
import sys

# log path
log_file = '/home/student/e21/e215706/dm/sorce/white/log/resnet34_log.txt'
os.makedirs(os.path.dirname(log_file), exist_ok=True)
sys.stdout = open(log_file, 'w')

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
# 3. Hyperparameter & csv, model path
# =======================
train_csv = '/home/student/e21/e215706/dm/sorce/image/csv/all_train.csv'
test_csv = '/home/student/e21/e215706/dm/sorce/image/csv/all_test.csv'
image_root = '/home/student/e21/e215706/dm/sorce/white/FaceDetection/all_image'

batch_size = 32
num_epochs = 10
learning_rate = 1e-4

# =======================
# 4. DataLoader
# =======================
train_dataset = LooksDataset(train_csv, image_root, transform)
test_dataset = LooksDataset(test_csv, image_root, transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# =======================
# 5. Model定義
# =======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet34(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

# =======================
# 6. Loss & 最適化
# =======================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# =======================
# 7. 学習Loop
# =======================
for epoch in range(num_epochs):
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
torch.save(model.state_dict(), '/home/student/e21/e215706/dm/sorce/white/model/resnet34_looks_classifier.pth')
print("saved model: resnet34_looks_classifier.pth")

sys.stdout.close()
