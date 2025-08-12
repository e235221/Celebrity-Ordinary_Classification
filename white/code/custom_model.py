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
log_file = root / paths["logs_dir"] / "resnet18_log.txt"
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
model = train(model, train_loader, val_loader, device,
              epochs=tr["num_epochs"], lr=tr["learning_rate"])

# 6) save
save_path = root / paths["models_create_dir"] / "resnet18_looks_classifier.pth"
save_model(model, save_path)