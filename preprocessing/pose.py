import os
import torch
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
from hopenet import Hopenet

# パス設定
image_dir = "/home/student/e21/e215706/dm/fix/info3dm_racial_classification/images_face_resized_300x300_fix_bbox" # 顔写真のパス
csv_path = "/home/student/e21/e215706/dm/fix/info3dm_racial_classification/test_imgs.csv" # ラベルcsvのパス
output_csv_path = "/home/student/e21/e215706/dm/fix/info3dm_racial_classification/test_imgs_pose.csv" # 出力csvパス
model_path = "/home/student/e21/e215706/dm/info3dm_racial_classification/pose/model/hopenet_alpha2.pkl" # モデルパス

# デバイス設定（GPUがあれば使用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# モデル読み込み
model = Hopenet(num_bins=66)
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict, strict=False)
model.to(device)
model.eval()

# 画像前処理定義
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Yaw 推定 + 分類
def get_yaw(image_path):
    img = Image.open(image_path).convert('RGB')
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        yaw, pitch, roll = model(tensor)
        bins_tensor = torch.arange(66).float().to(device)
        yaw_deg = torch.sum(torch.softmax(yaw, dim=1) * bins_tensor, dim=1) * 3 - 99
    return yaw_deg.item()

def classify_yaw(yaw):
    return "front" if abs(yaw) < 25 else "side"

# 実行
df = pd.read_csv(csv_path)
pose_labels = []

for i, row in df.iterrows():
    relative_path = row['file']
    full_path = os.path.join(image_dir, relative_path)
    try:
        yaw = get_yaw(full_path)
        pose = classify_yaw(yaw)
    except Exception as e:
        print(f"Error in {full_path}: {e}")
        pose = "error"
    pose_labels.append(pose)

# 保存
df['pose'] = pose_labels
df.to_csv(output_csv_path, index=False)
print(f"saved: {output_csv_path}")

