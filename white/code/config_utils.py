from pathlib import Path
import yaml

def _find_project_root(start: Path, target: str = "config.yaml") -> Path:
    p = start
    for parent in [p] + list(p.parents):
        if (parent / target).exists():
            return parent
    raise FileNotFoundError(f"Could not find {target} upwards from {start}")

def get_white_root() -> Path:
    return Path(__file__).resolve().parents[1]  # .../white

def load_cfg():
    white_root = get_white_root()
    project_root = _find_project_root(white_root)
    cfg_path = project_root / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return white_root, cfg

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)