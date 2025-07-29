import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ====== 1. 共通設定 ======
model_path = "/home/student/e21/e215706/dm/sorce/remove_back/model/custom.pth"
base_image_path = "/home/student/e21/e215706/dm/sorce/remove_back/all_image"
base_output_path = "/home/student/e21/e215706/dm/sorce/remove_back/analysis/gradcam_results/custom"
target_folders = ["good_test", "normal_test"]

# ====== 2. 画像前処理の定義 ======
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

to_rgb = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ====== 3. モデルの読み込み ======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet34(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device).eval()

# ====== 4. Grad-CAMの設定 ======
target_layer = model.layer4[-1]
cam = GradCAM(model=model, target_layers=[target_layer])

# ====== 5. 各フォルダに対してGrad-CAMを実行 ======
for folder in target_folders:
    image_folder = os.path.join(base_image_path, folder)
    output_folder = os.path.join(base_output_path, folder)
    os.makedirs(output_folder, exist_ok=True)

    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    print(f"処理中のフォルダ: {folder}（画像数: {len(image_files)}）")

    for img_file in tqdm(image_files, desc=f"Grad-CAM {folder}"):
        img_path = os.path.join(image_folder, img_file)
        img = Image.open(img_path).convert("RGBA")
        rgba = np.array(img)
        rgb = rgba[..., :3]
        alpha = rgba[..., 3:] / 255.0
        masked_rgb = (rgb * alpha).astype(np.uint8)
        masked_pil = Image.fromarray(masked_rgb).convert("RGB")

        input_tensor = transform(masked_pil).unsqueeze(0).to(device)
        rgb_tensor = to_rgb(masked_pil).permute(1, 2, 0).numpy()
        min_val = rgb_tensor.min()
        max_val = rgb_tensor.max()

        if max_val > min_val:
            rgb_tensor = (rgb_tensor - min_val) / (max_val - min_val)
        else:
            rgb_tensor = np.zeros_like(rgb_tensor)

        # Grad-CAMの適用
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0]
        cam_image = show_cam_on_image(rgb_tensor, grayscale_cam, use_rgb=True)

        # 保存
        out_path = os.path.join(output_folder, f"cam_{img_file}")
        cv2.imwrite(out_path, cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))

print("Grad-CAMの処理が完了しました。")
