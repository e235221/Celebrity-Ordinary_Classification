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
from torchvision.models import ResNet18_Weights

# 1. Path
model_path = "/home/student/e21/e215706/dm/sorce/model/create/resnet18_looks_classifier.pth"
image_folder = "/home/student/e21/e215706/dm/sorce/image/all_image/normal_test"
output_folder = "/home/student/e21/e215706/dm/sorce/analysis/gradcam_results/resnet18/normal_test"
os.makedirs(output_folder, exist_ok=True)

# 2. Image前処理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

to_rgb = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# 3. Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device).eval()

# 4. Grad-CAM
target_layer = model.layer4[-1]
cam = GradCAM(model=model, target_layers=[target_layer])

# 5. Image処理 Loop
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

for img_file in tqdm(image_files, desc="Grad-CAM processing"):
    img_path = os.path.join(image_folder, img_file)
    pil_img = Image.open(img_path).convert('RGB')
    
    input_tensor = transform(pil_img).unsqueeze(0).to(device)
    rgb_tensor = to_rgb(pil_img).permute(1, 2, 0).numpy()
    rgb_tensor = (rgb_tensor - rgb_tensor.min()) / (rgb_tensor.max() - rgb_tensor.min())  # normalize to 0-1

    # Grad-CAM適用
    targets = None
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
    cam_image = show_cam_on_image(rgb_tensor, grayscale_cam, use_rgb=True)

    # 保存
    out_path = os.path.join(output_folder, f"cam_{img_file}")
    cv2.imwrite(out_path, cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))

