"""
gradcam.py

このモジュールは、PyTorch と pytorch-grad-cam を利用して Grad-CAM を計算・保存するためのユーティリティ関数群を提供します。
主な機能は以下の通りです：
    - 学習済みモデルの構築（Grad-CAM 対応層の特定）
    - 単一画像またはフォルダ内の全画像に対する Grad-CAM 実行
    - フォルダ単位での Grad-CAM ヒートマップ平均化と保存
    - 入力画像の前処理（正規化、リサイズなど）

対象モデル：
    ResNet18 / ResNet34 / ResNet50 / EfficientNet-B0

依存ライブラリ：
    pytorch-grad-cam, torchvision, matplotlib, numpy, PIL
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import os
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

def make_cam_transforms(img_size: int = 224):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    to_rgb = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    return transform, to_rgb

def build_model_for_cam(arch: str, num_classes: int, model_path: Path, device: torch.device):
    arch = arch.lower()
    if arch == "resnet18":
        m = models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        target_layer = m.layer4[-1]
    elif arch == "resnet34":
        m = models.resnet34(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        target_layer = m.layer4[-1]
    elif arch == "resnet50":
        m = models.resnet50(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        target_layer = m.layer4[-1]
    elif arch in ("efficientnet_b0", "efficientnetb0"):
        m = models.efficientnet_b0(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        target_layer = m.features[-1]
    else:
        raise ValueError(f"Unsupported arch for Grad-CAM: {arch}")

    state = torch.load(model_path, map_location=device)
    m.load_state_dict(state)
    m.to(device).eval()
    return m, target_layer

def run_gradcam_on_folders(
    arch: str,
    model_path: Path,
    image_root: Path,
    out_root: Path,
    folders: List[str],
    num_classes: int = 2,
    img_size: int = 224
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, target_layer = build_model_for_cam(arch, num_classes, model_path, device)
    cam = GradCAM(model=model, target_layers=[target_layer])

    transform, to_rgb = make_cam_transforms(img_size)
    exts = (".jpg", ".jpeg", ".png")

    for folder in folders:
        src_dir = image_root / folder
        out_dir = out_root / folder
        out_dir.mkdir(parents=True, exist_ok=True)

        if not src_dir.exists():
            print(f"[WARN] skip (not found): {src_dir}")
            continue

        files = [f for f in os.listdir(src_dir) if f.lower().endswith(exts)]
        print(f"[INFO] {folder}: {len(files)} files")

        for fname in files:
            img_path = src_dir / fname
            pil_img = Image.open(img_path).convert("RGB")

            input_tensor = transform(pil_img).unsqueeze(0).to(device)

            rgb = to_rgb(pil_img).permute(1, 2, 0).numpy()
            vmin, vmax = rgb.min(), rgb.max()
            if vmax > vmin:
                rgb = (rgb - vmin) / (vmax - vmin)

            grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0]
            cam_img = show_cam_on_image(rgb, grayscale_cam, use_rgb=True)

            import cv2
            out_path = out_dir / f"cam_{fname}"
            cv2.imwrite(str(out_path), cv2.cvtColor(cam_img, cv2.COLOR_RGB2BGR))

    print(f"[DONE] Grad-CAM ({arch}) saved to: {out_root}")

def average_gradcam_on_folders(
    arch: str,
    model_path: Path,
    image_root: Path,
    out_dir: Path,
    folders: List[str],
    num_classes: int = 2,
    img_size: int = 224
):
    import matplotlib.pyplot as plt
    from tqdm import tqdm

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, target_layer = build_model_for_cam(arch, num_classes, model_path, device)
    cam = GradCAM(model=model, target_layers=[target_layer])

    transform, _ = make_cam_transforms(img_size)
    out_dir.mkdir(parents=True, exist_ok=True)

    heatmaps: Dict[str, np.ndarray] = {}
    counts: Dict[str, int] = {}

    for folder in folders:
        folder_path = image_root / folder
        files = [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg",".jpeg",".png"))]

        heatmap_sum = np.zeros((img_size, img_size), dtype=np.float32)
        cnt = 0

        print(f"processing: {folder} (n={len(files)})")
        for fname in tqdm(files, desc=f"Grad-CAM {folder}"):
            pil_img = Image.open(folder_path / fname).convert("RGB")
            input_tensor = transform(pil_img).unsqueeze(0).to(device)
            grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0]
            heatmap_sum += grayscale_cam.astype(np.float32)
            cnt += 1

        counts[folder] = max(cnt, 1)
        heatmaps[folder] = heatmap_sum

        avg_map = heatmap_sum / counts[folder]
        _save_avg_map(avg_map, out_dir / f"gradcam_{arch}_average_{folder}.png")

    total = sum(heatmaps[f] for f in folders) if folders else None
    total_cnt = sum(counts[f] for f in folders) if folders else 0
    if total is not None and total_cnt > 0:
        avg_all = total / total_cnt
        _save_avg_map(avg_all, out_dir / f"gradcam_{arch}_average_all.png")

def _save_avg_map(avg_map: np.ndarray, out_path: Path):
    import matplotlib.pyplot as plt
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 6))
    plt.imshow(avg_map, cmap="jet")
    plt.colorbar()
    plt.title(out_path.stem.replace("_", " "))
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"saved: {out_path}")
