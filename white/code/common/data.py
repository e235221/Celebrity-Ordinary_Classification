"""
data.py

このモジュールはデータローディング関連の機能をまとめています。
主な機能:
- LooksDataset: CSVファイルと画像ディレクトリからデータセットを構築する PyTorch Dataset クラス
- make_transforms: 画像サイズ変更・テンソル化・正規化を行う torchvision.transforms の定義
- make_loaders: 訓練用・評価用 DataLoader の生成

想定される用途:
モデル学習スクリプトから import して、共通のデータ前処理・ローダー生成処理を利用するために使います。
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class LooksDataset(Dataset):
    def __init__(self, csv_file: Path | str, image_root: Path | str, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_root = Path(image_root)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        row = self.data.iloc[idx]
        img_path = self.image_root / row["file"]
        image = Image.open(img_path).convert("RGB")
        label = int(row["looks"])
        if self.transform:
            image = self.transform(image)
        return image, label

def make_transforms(img_size: int = 224,
                    mean: Optional[list[float]] = None,
                    std: Optional[list[float]] = None):
    mean = mean or [0.5, 0.5, 0.5]
    std  = std  or [0.5, 0.5, 0.5]
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

def make_loaders(train_csv: Path | str,
                 test_csv: Path | str,
                 image_root: Path | str,
                 batch_size: int = 32,
                 num_workers: int = 2,
                 transform=None) -> Tuple[DataLoader, DataLoader]:
    transform = transform or make_transforms()
    train_ds = LooksDataset(train_csv, image_root, transform)
    test_ds  = LooksDataset(test_csv,  image_root, transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader