from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os

class LooksDataset(Dataset):
    def __init__(self, csv_file, img_root, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_root = img_root
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_root, self.data.iloc[idx]['file'])
        image = Image.open(img_path).convert('RGB')
        label = int(self.data.iloc[idx]['looks'])

        if self.transform:
            image = self.transform(image)

        return image, label

