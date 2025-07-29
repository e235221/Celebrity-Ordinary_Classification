from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import LooksDataset

def get_loader(csv_path, img_root, batch_size=32, shuffle=True):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    dataset = LooksDataset(csv_file=csv_path, img_root=img_root, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader

