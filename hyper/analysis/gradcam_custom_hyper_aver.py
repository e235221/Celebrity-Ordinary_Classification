"""
概要:
    Hyper 設定で学習した ResNet34(2クラス) を用い、white/org データの
    good_test / normal_test 各フォルダに対して Grad-CAM を計算し、
    フォルダ単位でヒートマップを平均化して保存するスクリプト。

主な処理:
    - config.yaml の読み込み（org をルートとしてパス解決）
    - 学習済みモデル（sorce/hyper/model/custom_hyper.pth）のロード
    - good_test / normal_test 全画像に対する Grad-CAM 実行
    - 各フォルダの平均ヒートマップと、全体平均(両フォルダ合算)の保存
      （出力先: sorce/hyper/analysis/）

入出力/副作用:
    - 入力: 画像ディレクトリ（sorce/org/image/all_image/**）,
            学習済みモデル（sorce/hyper/model/custom_hyper.pth）
    - 出力: 平均 Grad-CAM 画像（sorce/hyper/analysis/ に PNG 保存）
    - ログ: 標準出力に進捗表示

依存関係:
    - config_utils（パス解決/ディレクトリ作成）
    - torchvision, pytorch-grad-cam, numpy, PIL, matplotlib, tqdm, torch

実行例:
    python3 hyper/analysis/gradcam_custom_hyper_aver.py
"""

from pathlib import Path
import sys
import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
from pytorch_grad_cam import GradCAM
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[2] / "org" / "code"))
from config_utils import load_cfg, ensure_dir

# 1) path
org_root, cfg = load_cfg()                # …/sorce/org
project_root  = org_root.parent           # …/sorce
hyper_root    = project_root / "hyper"    # …/sorce/hyper

paths = cfg["paths"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_base  = org_root / paths["image_root"]                 # …/sorce/org/image/all_image
output_base = hyper_root / "analysis"                        # …/sorce/hyper/analysis
model_path  = hyper_root / "model" / "custom_hyper.pth"      # …/sorce/hyper/model/custom_hyper.pth
target_folders = ["good_test", "normal_test"]
ensure_dir(output_base)

# 2) Preprocessing
IMG_SIZE = 224
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3),
])

# 3) Model
model = models.resnet34(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
state = torch.load(model_path, map_location=device)
model.load_state_dict(state, strict=False)
model.to(device).eval()

# 4) Grad-CAM
target_layer = model.layer4[-1]
cam = GradCAM(model=model, target_layers=[target_layer])

# 5) Average (good and normal)
heatmaps, counts = {}, {}
for folder in target_folders:
    folder_path = image_base / folder
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    heatmap_sum = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
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
    out = output_base / f"gradcam_custom_hyper_average_{folder}.png"
    plt.figure(figsize=(6, 6))
    plt.imshow(avg_heatmap, cmap='jet')
    plt.colorbar(); plt.title(f"Average Grad-CAM(Hyper): {folder}")
    plt.axis('off'); plt.tight_layout()
    plt.savefig(out, dpi=300); plt.close()
    print(f"saved: {out}")

# 6) Average (all)
total = sum(heatmaps[f] for f in target_folders)
total_count = sum(counts[f] for f in target_folders)
avg_all = total / max(total_count, 1)

out_all = output_base / "gradcam_custom_hyper_average_all.png"
plt.figure(figsize=(6, 6))
plt.imshow(avg_all, cmap='jet')
plt.colorbar(); plt.title("Average Grad-CAM(Hyper): All (good + normal)")
plt.axis('off'); plt.tight_layout()
plt.savefig(out_all, dpi=300); plt.close()
print(f"saved: {out_all}")
