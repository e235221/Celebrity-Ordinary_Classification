"""
gradcam_resnet50_aver.py

ResNet50 モデルを用いてテスト用フォルダに含まれる画像に Grad-CAM を適用し、
クラスごとの Grad-CAM マップを平均化して保存するスクリプト。

機能概要:
- config.yaml からパス設定を読み込み
- 学習済み ResNet50 モデルをロード
- 指定されたフォルダ（例: good_test, normal_test）の全画像に対し Grad-CAM を実行
- クラスごとの Grad-CAM を平均化し、結果画像を out_dir に保存

想定用途:
- テストデータセット全体の特徴可視化傾向を確認するための分析
"""

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[2] / "code"))
from config_utils import load_cfg
from common.gradcam import average_gradcam_on_folders

root, cfg = load_cfg()
paths = cfg["paths"]

arch = "resnet50"
model_path = root / paths["models_dir"] / "resnet50_looks_classifier.pth"
image_root = root / paths["image_root"]
out_dir    = root / paths["gradcam_dir"] / "average"
folders    = ["good_test", "normal_test"]

average_gradcam_on_folders(arch, model_path, image_root, out_dir, folders, num_classes=2, img_size=224)