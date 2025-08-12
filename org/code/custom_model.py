"""
概要:
    ResNet34 ベースのカスタム学習スクリプト。
    外部チェックポイント（paths.pretrained_ckpt）が存在すれば strict=False でロード後、
    最終層を 2 クラス用に付け替えて学習する。

主な処理:
    - DataLoader 構築（common.data）
    - モデル構築（ResNet34 → 18クラス仮ヘッド → 外部CKPTロード → 最終fc=2に差し替え）
    - 学習/評価/保存（common.train）

入出力/副作用:
    - 入力: config.yaml（paths.*, train.*）, 外部CKPT（任意）
    - 出力: models_create_dir/custom.pth
    - ログ: logs_dir/custom_log.txt

依存関係:
    - config_utils, common.*
    - torchvision (resnet34)
    - torch.load(ckpt, strict=False) による部分ロード

実行例:
    python3 org/code/custom_model.py
"""

from pathlib import Path
import torch
import sys
sys.path.append(str(Path(__file__).parent))

from config_utils import load_cfg
from common.data import make_transforms, make_loaders
from common.models import build_model
from common.train import setup_logging, train, save_model

# 1) config
root, cfg = load_cfg()
paths, tr = cfg["paths"], cfg["train"]

# 2) logging
log_file = root / paths["logs_dir"] / "custom_log.txt"
setup_logging(log_file)

# 3) dataloaders
transform = make_transforms(img_size=224, mean=[0.5]*3, std=[0.5]*3)
train_loader, val_loader = make_loaders(
    train_csv=root / paths["train_csv"],
    test_csv =root / paths["test_csv"],
    image_root=root / paths["image_root"],
    batch_size=tr["batch_size"],
    num_workers=tr["num_workers"],
    transform=transform
)

# 4) model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_model("resnet34", num_classes=18, pretrained=tr.get("use_pretrained_imagenet", False))

ckpt_path = root / paths.get("pretrained_ckpt", "")
if ckpt_path.exists():
    state = torch.load(ckpt_path, map_location=device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded checkpoint: {ckpt_path}")
    if missing:   print(f"  missing keys: {len(missing)} (ok)")
    if unexpected:print(f"  unexpected keys: {len(unexpected)} (ok)")
else:
    print(f"Pretrained checkpoint not found (skip): {ckpt_path}")

import torch.nn as nn
model.fc = nn.Linear(model.fc.in_features, 2)

# 5) train
model = train(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
    epochs=tr["num_epochs"],
    lr=tr["learning_rate"],
)

# 6) save
save_model(model, root / paths["models_create_dir"] / "custom.pth")