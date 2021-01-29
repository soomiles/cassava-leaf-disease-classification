import pytorch_lightning as pl
from typing import Tuple, List, Optional

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from catalyst.data.sampler import BalanceClassSampler

from dataset.cassava import CassavaDataset


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_data_dir: str,
        val_data_dir: str,
        test_data_dir: str,
        submit_df_path: str,
        train_dataset_conf: Optional[DictConfig] = None,
        val_dataset_conf: Optional[DictConfig] = None,
        test_dataset_conf: Optional[DictConfig] = None,
        train_dataloader_conf: Optional[DictConfig] = None,
        val_dataloader_conf: Optional[DictConfig] = None,
        test_dataloader_conf: Optional[DictConfig] = None,
        df: pd.DataFrame = None,
        fold_num: int = None,
    ):
        super().__init__()
        self.train_data_dir = train_data_dir
        self.val_data_dir = val_data_dir
        self.test_data_dir = test_data_dir
        self.submit_df_path = submit_df_path
        self.train_dataset_conf = train_dataset_conf or OmegaConf.create()
        self.val_dataset_conf = val_dataset_conf or OmegaConf.create()
        self.test_dataset_conf = test_dataset_conf or OmegaConf.create()
        self.train_dataloader_conf = train_dataloader_conf or OmegaConf.create()
        self.val_dataloader_conf = val_dataloader_conf or OmegaConf.create()
        self.test_dataloader_conf = test_dataloader_conf or OmegaConf.create()
        self.df = df
        self.fold_num = fold_num if fold_num is not None else 0

    def setup(self, stage: Optional[str] = None):
        train_df = self.df[self.df.fold != self.fold_num].reset_index(drop=True)
        val_df = self.df[self.df.fold == self.fold_num].reset_index(drop=True)
        if stage == "fit" or stage is None:
            self.train = CassavaDataset(self.train_data_dir, train_df, train=True,
                                        **self.train_dataset_conf)
            self.val = CassavaDataset(self.val_data_dir, val_df, train=False,
                                      **self.val_dataset_conf)

        if stage == "test" or stage is None:
            self.val = CassavaDataset(self.val_data_dir, val_df, train=False,
                                      **self.val_dataset_conf)

    def train_dataloader(self):
        return DataLoader(self.train, **self.train_dataloader_conf)

    def val_dataloader(self):
        return DataLoader(self.val, **self.val_dataloader_conf)

    def test_dataloader(self):
        return DataLoader(self.val, **self.val_dataloader_conf)