from pathlib import Path
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset
from torch.nn.functional import one_hot

KERMANY_LABELS = {'CNV': 1, 'DME': 2, 'DRUSEN': 3, 'NORMAL': 0}

class KermanyDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        self.imgs = []

        img_dir = Path(img_dir)
        for label_dir in img_dir.iterdir():
            label = label_dir.name
            for image_path in Path(label_dir).iterdir():
                self.imgs.append((image_path, one_hot(torch.tensor(KERMANY_LABELS[label]), 4)))

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        image_path, label = self.imgs[idx]
        image = read_image(str(image_path))

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        image = torch.stack([image, image, image]).squeeze()

        return image, label
