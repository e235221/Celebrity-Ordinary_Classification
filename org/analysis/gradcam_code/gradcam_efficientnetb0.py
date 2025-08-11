import os
from pathlib import Path
import sys
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import models, transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# ====== 1. 共通設定 ======
sys.path.append(str(Path(__file__).resolve().parents[2] / "code"))
from config_utils import load_cfg, ensure_dir
root, cfg = load_cfg()
paths = cfg["paths"]
models_dir   = root / paths["models_dir"]
image_root   = root / paths["image_root"]
gradcam_root = root / paths["gradcam_dir"] / "efficientnetb0"
model_path = models_dir / "efficientnetb0_looks_classifier.pth"
target_folders = ["normal_test", "good_test"]

# ====== 2. 画像前処理 ======
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])
to_rgb = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ====== 3. モデル読み込み ======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
state = torch.load(model_path, map_location=device)
model.load_state_dict(state)
model.to(device).eval()

# ========== 4) Grad-CAM ==========
target_layer = model.features[-1]
cam = GradCAM(model=model, target_layers=[target_layer])

# ====== 5. 処理 ======
exts = (".jpg", ".jpeg", ".png")
for folder in target_folders:
    src_dir = image_root / folder
    out_dir = gradcam_root / folder
    ensure_dir(out_dir)

    if not src_dir.exists():
        print(f"[WARN] skip (not found): {src_dir}")
        continue

    image_files = [f for f in os.listdir(src_dir) if f.lower().endswith(exts)]
    print(f"[INFO] {folder}: {len(image_files)} files")

    for fname in tqdm(image_files, desc=f"Grad-CAM {folder}"):
        img_path = src_dir / fname
        pil_img = Image.open(img_path).convert("RGB")

        input_tensor = transform(pil_img).unsqueeze(0).to(device)

        rgb = to_rgb(pil_img).permute(1, 2, 0).numpy()
        vmin, vmax = rgb.min(), rgb.max()
        if vmax > vmin:
            rgb = (rgb - vmin) / (vmax - vmin)

        grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0]
        cam_img = show_cam_on_image(rgb, grayscale_cam, use_rgb=True)

        out_path = out_dir / f"cam_{fname}"
        cv2.imwrite(str(out_path), cv2.cvtColor(cam_img, cv2.COLOR_RGB2BGR))

print("[DONE] Grad-CAM for efficientnetb0 on normal/good test")