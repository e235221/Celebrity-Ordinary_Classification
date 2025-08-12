from pathlib import Path
import sys, time
import torch
import torch.nn as nn

ORG_CODE = Path(__file__).resolve().parents[1] / "org" / "code"
sys.path.append(str(ORG_CODE))

from config_utils import load_cfg, ensure_dir
from common.data import make_transforms, make_loaders
from common.models import build_model
from common.train import setup_logging, train, save_model

# 1) Hyper Parameter
EPOCH   = 5
BATCH   = 64
WORKERS = 4

# 2) Path
org_root, cfg = load_cfg()                 # org (/.../sorce/org), config load
project_root  = org_root.parent            # /.../sorce
hyper_root    = project_root / "hyper"     # /.../sorce/hyper

paths, tr = cfg["paths"], cfg["train"]

log_file = hyper_root / "log" / "custom_hyper_log.txt"
ensure_dir(log_file.parent)
setup_logging(log_file)

train_csv  = org_root / paths["train_csv"]
test_csv   = org_root / paths["test_csv"]
image_root = org_root / paths["image_root"]

model_out = hyper_root / "model" / "custom_hyper.pth"
ensure_dir(model_out.parent)

ckpt_path = org_root / paths.get("pretrained_ckpt", "")

# 3) DataLoader
transform = make_transforms(img_size=224, mean=[0.5]*3, std=[0.5]*3)
train_loader, val_loader = make_loaders(
    train_csv=train_csv,
    test_csv=test_csv,
    image_root=image_root,
    batch_size=BATCH or tr["batch_size"],
    num_workers=WORKERS or tr["num_workers"],
    transform=transform,
)

# 4) Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_model("resnet34", num_classes=18, pretrained=tr.get("use_pretrained_imagenet", False)).to(device)

if ckpt_path.exists():
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state, strict=False)
    print(f"Loaded checkpoint: {ckpt_path}")
else:
    print(f"Pretrained checkpoint not found (skip): {ckpt_path}")

model.fc = nn.Linear(model.fc.in_features, 2).to(device)

# 5) Train
start = time.time()
model = train(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
    epochs=EPOCH,
    lr=tr["learning_rate"],
)
# 6) Save & Time
save_model(model, model_out)
elapsed_min = (time.time() - start) / 60
print(f"Total Training Time: {elapsed_min:.2f} minutes")
