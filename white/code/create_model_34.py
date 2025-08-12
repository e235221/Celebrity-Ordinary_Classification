"""
概要:
    ResNet34 を用いた 2 クラス分類モデルの学習スクリプト。
    config.yaml の設定に基づき、共通モジュールを利用して学習を実行する。

主な処理:
    - Dataset/DataLoader 構築（common.data）
    - モデル構築（ResNet34, common.models）
    - 学習/評価/保存（common.train）

入出力/副作用:
    - 入力: config.yaml
    - 出力: models_create_dir/resnet34_looks_classifier.pth
    - ログ: logs_dir/resnet34_log.txt

依存関係:
    - config_utils, common.data, common.models, common.train
    - torchvision (resnet34)

実行例:
    python3 white/code/create_model_34.py
"""

from pathlib import Path
import sys
import torch
sys.path.append(str(Path(__file__).parent))

from config_utils import load_cfg
from common.data import make_transforms, make_loaders
from common.models import build_model
from common.train import setup_logging, train, save_model

# 1) config
root, cfg = load_cfg()
paths, tr = cfg["paths"], cfg["train"]

# 2) logging
log_file = root / paths["logs_dir"] / "resnet34_log.txt"
setup_logging(log_file)

# 3) dataloaders
org_root   = root.parent / "org"    # sorce/org
train_csv  = org_root / "image/csv/all_train.csv"
test_csv   = org_root / "image/csv/all_test.csv"
image_root = root / paths["image_root"]   # sorce/white/image/all_image

transform = make_transforms(img_size=224, mean=[0.5]*3, std=[0.5]*3)
train_loader, val_loader = make_loaders(
    train_csv=train_csv,
    test_csv=test_csv,
    image_root=image_root,
    batch_size=tr["batch_size"],
    num_workers=tr["num_workers"],
    transform=transform
)

# 4) model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_model("resnet34", num_classes=2, pretrained=tr.get("use_pretrained_imagenet", False))

# 5) train
model = train(model, train_loader, val_loader, device,
              epochs=tr["num_epochs"], lr=tr["learning_rate"])

# 6) save
save_path = root / paths["models_create_dir"] / "resnet34_looks_classifier.pth"
save_model(model, save_path)