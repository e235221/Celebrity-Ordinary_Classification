from pathlib import Path
import yaml

def _find_project_root(start: Path, target: str = "config.yaml") -> Path:
    p = start
    for parent in [p] + list(p.parents):
        if (parent / target).exists():
            return parent
    raise FileNotFoundError(f"Could not find {target} upwards from {start}")

def get_org_root() -> Path:
    return Path(__file__).resolve().parents[1]  # .../org

def load_cfg():
    org_root = get_org_root()                # /.../sorce/org
    project_root = _find_project_root(org_root)  # /.../sorce
    cfg_path = project_root / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return org_root, cfg

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)