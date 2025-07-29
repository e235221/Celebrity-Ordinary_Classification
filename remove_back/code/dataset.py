from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os
import torch
import numpy as np

class LooksDataset(Dataset):
    def __init__(self, csv_file, img_root, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_root = img_root
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_root, self.data.iloc[idx]['file'])
        img = Image.open(img_path).convert("RGBA")
        rgba = np.array(img)
        rgb = rgba[..., :3]
        alpha = rgba[..., 3]
        alpha_mask = (alpha / 255.0).astype(np.float32)[..., np.newaxis]
        masked_rgb = rgb * alpha_mask
        masked_img = Image.fromarray(masked_rgb.astype(np.uint8))
        if self.transform:
            masked_img = self.transform(masked_img)
        label = int(self.data.iloc[idx]['looks'])
        return masked_img, label

