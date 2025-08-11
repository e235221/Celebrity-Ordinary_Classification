from pathlib import Path
import yaml

def get_root() -> Path:
    return Path(__file__).resolve().parents[1]

def load_cfg():
    root = get_root()
    with open(root / "config.yaml", "r", encoding="utf-8") as f:
        return root, yaml.safe_load(f)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
