import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
from torchvision import transforms, models
import torch.nn as nn
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
import sys

# ====== 1. 共通設定 ======
sys.path.append(str(Path(__file__).resolve().parents[2] / "code"))
from config_utils import load_cfg, ensure_dir
root, cfg = load_cfg()
paths = cfg["paths"]
models_dir  = root / paths["models_dir"]
image_base  = root / paths["image_root"]
output_base = root / paths["gradcam_dir"] / "average"
ensure_dir(output_base)
model_path = models_dir / "resnet34_looks_classifier.pth"
target_folders = ["good_test", "normal_test"]

# ====== 2. 画像前処理 ======
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])

# ====== 3. モデル読み込み ======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet34(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device).eval()

# ====== 4. Grad-CAM定義 ======
target_layer = model.layer4[-1]
cam = GradCAM(model=model, target_layers=[target_layer])

# ====== 5. 平均計算&出力 ======
heatmaps, counts = {}, {}
# normal, good
for folder in target_folders:
    folder_path = image_base / folder
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    heatmap_sum = np.zeros((224, 224), dtype=np.float32)
    count = 0

    print(f"processing: {folder} (n={len(image_files)})")
    for img_file in tqdm(image_files, desc=f"Grad-CAM {folder}"):
        img_path = folder_path / img_file
        image = Image.open(img_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0]
        heatmap_sum += grayscale_cam.astype(np.float32)
        count += 1

    counts[folder] = max(count, 1)
    heatmaps[folder] = heatmap_sum

    avg_heatmap = heatmap_sum / counts[folder]
    out = output_base / f"gradcam_resnet34_average_{folder}.png"
    plt.figure(figsize=(6, 6))
    plt.imshow(avg_heatmap, cmap='jet')
    plt.colorbar(); plt.title(f"Average Grad-CAM: {folder}")
    plt.axis('off'); plt.tight_layout()
    plt.savefig(out, dpi=300); plt.close()
    print(f"saved: {out}")

# normal+good
total = sum(heatmaps[f] for f in target_folders)
total_count = sum(counts[f] for f in target_folders)
avg_all = total / max(total_count, 1)

out_all = output_base / "gradcam_resnet34_average_all.png"
plt.figure(figsize=(6, 6))
plt.imshow(avg_all, cmap='jet')
plt.colorbar(); plt.title("Average Grad-CAM: All")
plt.axis('off'); plt.tight_layout()
plt.savefig(out_all, dpi=300); plt.close()
print(f"saved: {out_all}")
