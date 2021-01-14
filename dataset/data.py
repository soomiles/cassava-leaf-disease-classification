import pytorch_lightning as pl
from hydra.utils import to_absolute_path
from typing import Tuple, List, Optional
from omegaconf import DictConfig, OmegaConf

import os
import pandas as pd
import logging
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import instantiate
from pytorch_lightning.utilities.seed import seed_everything

from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from dataset.transforms import get_transforms
from dataset.cassava import CassavaDataset


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        data_conf: Optional[DictConfig] = None,
    ):
        super().__init__()
        self.train_df = train_df
        self.valid_df = valid_df
        self.data_conf = data_conf or OmegaConf.create()

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            tfms = get_transforms(img_size=self.data_conf.img_size)
            self.train = CassavaDataset(self.train_df, self.data_conf, transforms=tfms['train'])
            self.val = CassavaDataset(self.valid_df, self.data_conf, transforms=tfms['val'],
                                      train=False)

        if stage == "test" or stage is None:
            pass

    def train_dataloader(self):
        return DataLoader(self.train, **self.train_dataloader_conf)

    def val_dataloader(self):
        return DataLoader(self.val, **self.val_dataloader_conf)

    def test_dataloader(self):
        return DataLoader(self.test, **self.test_dataloader_conf)