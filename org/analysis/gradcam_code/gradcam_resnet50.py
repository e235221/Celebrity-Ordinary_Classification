"""
gradcam_resnet50.py

ResNet50 モデルを用いてテスト用フォルダに含まれる画像ごとに Grad-CAM を適用し、
各画像ごとの可視化結果を保存するスクリプト。

機能概要:
- config.yaml からパス設定を読み込み
- 学習済み ResNet50 モデルをロード
- 指定された複数フォルダ（例: normal_test, good_test）の全画像に対し Grad-CAM を実行
- 画像ごとの Grad-CAM 可視化結果を保存

想定用途:
- 個別画像単位での特徴可視化と、モデルが注目している領域の確認
"""

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[2] / "code"))
from config_utils import load_cfg
from common.gradcam import run_gradcam_on_folders

root, cfg = load_cfg()
paths = cfg["paths"]

arch = "resnet50"
model_path = root / paths["models_dir"] / "resnet50_looks_classifier.pth"
image_root = root / paths["image_root"]
out_root   = root / paths["gradcam_dir"] / arch
folders    = ["normal_test", "good_test"]

run_gradcam_on_folders(arch, model_path, image_root, out_root, folders, num_classes=2, img_size=224)