from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.io import read_image
from torch.utils.data import Dataset
from torch.nn.functional import one_hot
from torchvision import transforms

KERMANY_LABELS = {'NORMAL': torch.nn.functional.one_hot(torch.tensor(0), 4),
                  'CNV': torch.nn.functional.one_hot(torch.tensor(1), 4),
                  'DME': torch.nn.functional.one_hot(torch.tensor(2), 4),
                  'DRUSEN': torch.nn.functional.one_hot(torch.tensor(3), 4)}

mask_values = [10, 26, 51, 77, 102, 128, 153, 179, 204, 230]
fg_mask_values = [10, 77, 102, 128, 153, 179, 204, 230]


class KermanyDataset(Dataset):
    def __init__(self, img_dir, mask_dir, mode=0, image_transform=None, mask_transform=None, target_transform=None):
        self.imgs = []

        img_dir, mask_dir = Path(img_dir), Path(mask_dir)
        for label_dir in img_dir.iterdir():
            label = label_dir.name
            for image_path in Path(label_dir).glob('*.jpeg'):
                mask_path = mask_dir / image_path.parent.name / image_path.with_suffix('.bmp').name
                self.imgs.append((image_path, mask_path, KERMANY_LABELS[label]))

        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.target_transform = target_transform
        self.mode = mode

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        image_path, mask_path, label = self.imgs[idx]
        # image_path = '/home/projects/ronen/sgvdan/workspace/datasets/kermany/original/train/CNV/CNV-7393104-63.jpeg'
        # mask_path = '/home/projects/ronen/sgvdan/workspace/datasets/kermany/layer-segmentation/train/CNV/CNV-7393104-63.bmp'
        image = (transforms.ToTensor()(Image.open(str(image_path))) * 255).type(torch.uint8)
        mask = (transforms.ToTensor()(Image.open(str(mask_path))) * 255).type(torch.uint8)

        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        if self.target_transform:
            label = self.target_transform(label)

        if self.mode == 0:
            imt = image
        elif self.mode == 1:
            imt = torch.concat([mask, image], dim=0)
        elif self.mode == 2:
            channel_mask = torch.concat([(mask == value) for value in mask_values], dim=0).type(torch.uint8) * 255
            imt = torch.concat([channel_mask, image], dim=0)
        elif self.mode == 3:
            imt = torch.concat([torch.zeros_like(mask), image], dim=0)
        elif self.mode == 4:
            mask[mask == 26] = 0
            mask[mask == 51] = 0
            imt = torch.concat([mask, image], dim=0)
        elif self.mode == 5:
            fg_channel_mask = torch.concat([(mask == value) for value in fg_mask_values], dim=0).type(torch.uint8) * 255
            imt = torch.concat([fg_channel_mask, image], dim=0)
        elif 6 <= self.mode <= 15:
            value = mask_values[self.mode - 6]
            single_channel_mask = (mask == value).type(torch.uint8) * 255
            imt = torch.concat([single_channel_mask, image], dim=0)
        else:
            raise NotImplementedError

        return imt, label
