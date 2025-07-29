import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from PIL import Image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
import numpy as np

from dataset import LooksDataset

# Data Setting
test_csv = "/home/student/e21/e215706/dm/sorce/image/csv/all_test.csv"
image_root = "/home/student/e21/e215706/dm/sorce/image/all_image"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

test_dataset = LooksDataset(test_csv, image_root, transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model Setting
model = models.efficientnet_b0(pretrained=False)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
model.load_state_dict(torch.load("/home/student/e21/e215706/dm/sorce/model/create/efficientnetb0_looks_classifier.pth", map_location=device))
model = model.to(device)
model.eval()

# 予測
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

# 混同行列生成
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["normal(0)", "good(1)"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix (efficientnet_b0)")
plt.savefig("/home/student/e21/e215706/dm/sorce/analysis/confusion_results/confusion_matrix_efficientnetb0.png")  # 保存path
plt.show()
print(classification_report(all_labels, all_preds, target_names=["normal", "good"]))

