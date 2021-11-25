from pathlib import Path

import cv2
import numpy as np
import torch
from oct_converter.readers import E2E

from torch.utils.data import Dataset
import PIL
from skimage import exposure
import torchvision
from tqdm import tqdm

""" NOTE: This is just a sample dataset... it's not guaranteed that this will work"""


class E2ETileDataset(Dataset):
    def __init__(self, cache, transform):
        """
        Args:
            volumes: list of 3-d volumes (recommend providing paths and loading them on-the-fly if the dataset is large)
            labels: list of labels(arrays or int) for each volume
        """
        self.cache = cache
        self.transform = transform
        # Save all volumes to cache
        
    def __len__(self):
        return len(self.cache)

    def __getitem__(self, idx):
        volume, label = self.cache[idx]
        # volume is of shape (n_slices x H x W)

        # contrast stretch
        # contrast_stretch = pil_contrast_strech() TODO: they did this on original article. Think if we want to add it.
        slices = [torch.tensor(cv2.resize(slice, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)) for slice in volume]

        # assuming we had a single channel input, just concatenate them to make 3 channels
        v = torch.cat(slices, dim=0)
        # v is of shape ([n_slices * H] x W)
        v = torch.stack([v,v,v]).squeeze()
        # v is of shape(3 x [n_slices * H] x W)
        
        # return sample
        return v, label


class  pil_contrast_strech(object):
    def __init__(self,low=2,high=98):
        self.low,self.high = low,high

    def __call__(self,img):
        # Contrast stretching
        img=np.array(img)
        plow, phigh = np.percentile(img, (self.low, self.high))
        return PIL.Image.fromarray(exposure.rescale_intensity(img, in_range=(plow, phigh)))


def build_volume_cache(cache, path, label):
    """

    :param cache:
    :param path:
    :param label:
    :param limit:
    :param config:
    :return:
    """

    counter = 0
    for sample in tqdm(list(Path(path).rglob("*.E2E"))):
        if sample.is_file():
            for volume in E2E(sample).read_oct_volume():
                if isinstance(volume.volume[0], np.ndarray):
                    if len(volume.volume) > 16:  # at least 16 slices mandatory for UCLA's technique
                        try:
                            cache.append((volume.volume, label))
                            counter += 1
                        except Exception as ex:
                            print("Ignored volume {0} in sample {1}. An exception of type {2} occurred. \
                                       Arguments:\n{1!r}".format(volume.patient_id, sample, type(ex).__name__, ex.args))
                            continue
