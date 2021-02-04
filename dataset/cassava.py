import torch
import numpy as np
import pandas as pd
import cv2
from PIL import Image

from torch.utils.data import Dataset
from typing import Tuple, List, Optional
from omegaconf import DictConfig, OmegaConf
from ast import literal_eval

from dataset.transforms import get_transforms
from dataset.fmix import make_low_freq_image, binarise_mask
from dataset.randaugment import RandAugment


def get_img(path):
    im_bgr = cv2.imread(path)
    im_rgb = im_bgr[:, :, ::-1]
    return im_rgb

def rand_bbox(size, lam):
    W = size[0]
    H = size[1]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


class CassavaDataset(Dataset):
    def __init__(self,
                 img_dir: str,
                 df: pd.DataFrame,
                 train: bool,
                 img_size: str = None,
                 output_label: int = None,
                 one_hot_label: int = None,
                 do_randaug: bool = False,
                 randaug_params: Optional[DictConfig] = None,
                 do_fmix: int = None,
                 fmix_params: Optional[DictConfig] = None,
                 do_cutmix: int = None,
                 cutmix_params: Optional[DictConfig] = None,
                 ):

        super().__init__()
        self.img_dir = img_dir
        self.df = df.copy()
        img_size = literal_eval(img_size)
        self.h, self.w = img_size
        self.do_randaug = do_randaug
        self.randaug_params = randaug_params
        self.do_fmix = do_fmix
        self.fmix_params = fmix_params
        self.fmix_params.shape = img_size
        self.do_cutmix = do_cutmix
        self.cutmix_params = cutmix_params

        if self.do_randaug:
            self.randaug = RandAugment(**self.randaug_params)
        else:
            self.randaug = None

        if train:
            self.transforms = get_transforms(need=('train'),
                                             img_size=img_size,
                                             do_randaug=self.do_randaug)['train']
        else:
            self.transforms = get_transforms(need=('val'),
                                             img_size=img_size, crop=True)['val']

        self.output_label = output_label
        self.one_hot_label = one_hot_label

        if self.output_label == True:
            self.labels = self.df['label'].values

            if self.one_hot_label is True:
                if not isinstance(self.labels[0], str):
                    self.labels = np.eye(5)[self.labels]
                else:
                    self.labels = np.array([literal_eval(elem) for elem in self.labels])
        self.labels_copy = self.labels.copy()

    def get_classes(self):
        return self.df['label'].values

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index: int):

        # get labels
        if self.output_label:
            target = self.labels[index]

        img = get_img("{}/{}".format(self.img_dir, self.df.loc[index]['image_id']))

        if self.transforms:
            if self.do_randaug:
                img = np.array(self.randaug(Image.fromarray(img)))
            img = self.transforms(image=img)['image']

        if self.do_fmix and np.random.uniform(0., 1., size=1)[0] > 0.5:
            with torch.no_grad():
                lam = np.clip(np.random.beta(self.fmix_params['alpha'], self.fmix_params['alpha']), 0.6, 0.7)

                # Make mask, get mean / std
                mask = make_low_freq_image(self.fmix_params['decay_power'], self.fmix_params['shape'])
                mask = binarise_mask(mask, lam, self.fmix_params['shape'], self.fmix_params['max_soft'])

                fmix_ix = np.random.choice(self.df.index, size=1)[0]
                fmix_img = get_img("{}/{}".format(self.img_dir, self.df.iloc[fmix_ix]['image_id']))

                if self.transforms:
                    fmix_img = self.transforms(image=fmix_img)['image']

                mask_torch = torch.from_numpy(mask).float()

                # mix image
                img = mask_torch * img + (1. - mask_torch) * fmix_img

                rate = mask.sum() / self.h / self.w
                target = rate * target + (1. - rate) * self.labels[fmix_ix]

        if self.do_cutmix and np.random.uniform(0., 1., size=1)[0] > 0.5:
            with torch.no_grad():
                cmix_ix = np.random.choice(self.df.index, size=1)[0]
                cmix_img = get_img("{}/{}".format(self.img_dir, self.df.iloc[cmix_ix]['image_id']))
                if self.transforms:
                    if self.randaug:
                        cmix_img = np.array(self.randaug(Image.fromarray(cmix_img)))
                    cmix_img = self.transforms(image=cmix_img)['image']

                lam = np.clip(np.random.beta(self.cutmix_params['alpha'], self.cutmix_params['alpha']), 0.3, 0.4)
                bbx1, bby1, bbx2, bby2 = rand_bbox((self.h, self.w), lam)

                img[:, bbx1:bbx2, bby1:bby2] = cmix_img[:, bbx1:bbx2, bby1:bby2]

                rate = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (self.h * self.w))
                target = rate * target + (1. - rate) * self.labels[cmix_ix]

        if self.output_label == True:
            return img, target
        else:
            return img
