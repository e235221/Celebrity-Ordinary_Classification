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
log_file = root / paths["logs_dir"] / "efficientnetb0_log.txt"
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
model = build_model("efficientnetb0", num_classes=2, pretrained=tr.get("use_pretrained_imagenet", False))

# 5) train
model = train(model, train_loader, val_loader, device,
              epochs=tr["num_epochs"], lr=tr["learning_rate"])

# 6) save
save_model(model, root / paths["models_create_dir"] / "efficientnetb0_looks_classifier.pth")