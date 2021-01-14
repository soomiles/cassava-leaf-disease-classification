import torch
import numpy as np
import pandas as pd
import cv2

from torch.utils.data import Dataset
from omegaconf import DictConfig

from dataset.fmix import make_low_freq_image, binarise_mask


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
    def __init__(self, df: pd.DataFrame,
                 dataset_conf: DictConfig,
                 transforms=None,
                 train=True
                 ):

        super().__init__()
        self.df = df.reset_index(drop=True).copy()
        self.transforms = transforms
        self.img_dir = dataset_conf.img_dir
        self.img_size = dataset_conf.img_size
        self.do_fmix = dataset_conf.do_fmix
        self.fmix_params = dataset_conf.fmix_params
        self.do_cutmix = dataset_conf.do_cutmix
        self.cutmix_params = dataset_conf.cutmix_params

        self.output_label = dataset_conf.output_label
        self.one_hot_label = dataset_conf.one_hot_label

        if self.output_label == True:
            self.labels = self.df['label'].values

            if self.one_hot_label is True:
                self.labels = np.eye(self.df['label'].max() + 1)[self.labels]

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index: int):

        # get labels
        if self.output_label:
            target = self.labels[index]

        img = get_img("{}/{}".format(self.img_dir, self.df.loc[index]['image_id']))

        if self.transforms:
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

                mask_torch = torch.from_numpy(mask)

                # mix image
                img = mask_torch * img + (1. - mask_torch) * fmix_img

                rate = mask.sum() / self.img_size / self.img_size
                target = rate * target + (1. - rate) * self.labels[fmix_ix]

        if self.do_cutmix and np.random.uniform(0., 1., size=1)[0] > 0.5:
            with torch.no_grad():
                cmix_ix = np.random.choice(self.df.index, size=1)[0]
                cmix_img = get_img("{}/{}".format(self.img_dir, self.df.iloc[cmix_ix]['image_id']))
                if self.transforms:
                    cmix_img = self.transforms(image=cmix_img)['image']

                lam = np.clip(np.random.beta(self.cutmix_params['alpha'], self.cutmix_params['alpha']), 0.3, 0.4)
                bbx1, bby1, bbx2, bby2 = rand_bbox((self.img_size, self.img_size), lam)

                img[:, bbx1:bbx2, bby1:bby2] = cmix_img[:, bbx1:bbx2, bby1:bby2]

                rate = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (self.img_size * self.img_size))
                target = rate * target + (1. - rate) * self.labels[cmix_ix]

        if self.output_label == True:
            return img, target
        else:
            return img
