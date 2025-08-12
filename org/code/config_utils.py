"""
概要:
    プロジェクト共通の設定ローダー/ユーティリティ。
    org ディレクトリをルート(root)として扱い、リポジトリ直下の config.yaml を読み込む。

主な機能:
    - load_cfg(): org ルートと設定(dict)を返す
    - ensure_dir(path): ディレクトリが無ければ作成

入出力/副作用:
    - 入力: リポジトリ直下の config.yaml
    - 出力: なし（呼び出し側へ設定を返す）
    - 副作用: なし（ensure_dir はディレクトリ作成あり）

依存関係:
    - pathlib, yaml

使い方(例):
    from config_utils import load_cfg, ensure_dir
    root, cfg = load_cfg()
    ensure_dir(root / "log")
"""

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