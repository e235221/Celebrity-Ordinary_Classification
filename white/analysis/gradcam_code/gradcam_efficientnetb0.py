from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[2] / "code"))
from config_utils import load_cfg
from common.gradcam import run_gradcam_on_folders

root, cfg = load_cfg()
paths = cfg["paths"]

arch = "efficientnetb0"
model_path = root / paths["models_dir"] / "efficientnetb0_looks_classifier.pth"
image_root = root / paths["image_root"]
out_root   = root / paths["gradcam_dir"] / arch
folders    = ["normal_test", "good_test"]

run_gradcam_on_folders(arch, model_path, image_root, out_root, folders, num_classes=2, img_size=224)