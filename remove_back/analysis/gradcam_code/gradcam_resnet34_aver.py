import numpy as np
import cv2
import os
from tqdm import tqdm
from PIL import Image
import torch
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from torchvision import models
import torch.nn as nn
import matplotlib.pyplot as plt

# ====== 1. 共通設定 ======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_base = "/home/student/e21/e215706/dm/sorce/remove_back/all_image"
output_base = "/home/student/e21/e215706/dm/sorce/remove_back/analysis/gradcam_results/average"
model_path = "/home/student/e21/e215706/dm/sorce/remove_back/model/resnet34_looks_classifier.pth"
target_folders = ["good_test", "normal_test"]

# ====== 2. 画像前処理 ======
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])

# ====== 3. モデル読み込み ======
model = models.resnet34(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device).eval()

# ====== 4. Grad-CAM定義 ======
target_layer = model.layer4[-1]
cam = GradCAM(model=model, target_layers=[target_layer])

# ====== 5. 平均計算用の辞書 ======
heatmaps = {}
counts = {}

# ====== 6. 各フォルダ処理 ======
for folder in target_folders:
    folder_path = os.path.join(image_base, folder)
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    heatmap_sum = np.zeros((224, 224))
    count = 0

    print(f"処理中: {folder}（画像数: {len(image_files)}）")

    for img_file in tqdm(image_files, desc=f"Grad-CAM {folder}"):
        img_path = os.path.join(folder_path, img_file)

        # RGBA読み込みしてアルファマスク適用
        img = Image.open(img_path).convert("RGBA")
        rgba = np.array(img)
        rgb = rgba[..., :3]
        alpha = rgba[..., 3:] / 255.0  # (H, W, 1)
        masked_rgb = (rgb * alpha).astype(np.uint8)
        masked_pil = Image.fromarray(masked_rgb).convert("RGB")

        # モデル入力に変換
        input_tensor = transform(masked_pil).unsqueeze(0).to(device)
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0]
        heatmap_sum += grayscale_cam
        count += 1

    heatmaps[folder] = heatmap_sum
    counts[folder] = count

    # 保存: 個別フォルダの平均
    avg_heatmap = heatmap_sum / count
    output_path = os.path.join(output_base, f"gradcam_resnet34_average_{folder}_remove_back.png")
    plt.figure(figsize=(6, 6))
    plt.imshow(avg_heatmap, cmap='jet')
    plt.colorbar()
    plt.title(f"Average Grad-CAM: {folder}")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"保存完了: {output_path}")

# ====== 7. good + normal 合計ヒートマップ ======
total_heatmap = sum(heatmaps.values())
total_count = sum(counts.values())
avg_all = total_heatmap / total_count

output_all_path = os.path.join(output_base, "gradcam_resnet34_average_all_remove_back.png")
plt.figure(figsize=(6, 6))
plt.imshow(avg_all, cmap='jet')
plt.colorbar()
plt.title("Average Grad-CAM: All (good + normal)")
plt.axis('off')
plt.tight_layout()
plt.savefig(output_all_path, dpi=300)
plt.close()
print(f"保存完了: {output_all_path}")

