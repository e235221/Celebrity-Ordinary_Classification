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