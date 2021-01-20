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
        df_path: str,
        submit_df_path: str,
        train_dataset_conf: Optional[DictConfig] = None,
        val_dataset_conf: Optional[DictConfig] = None,
        test_dataset_conf: Optional[DictConfig] = None,
        train_dataloader_conf: Optional[DictConfig] = None,
        val_dataloader_conf: Optional[DictConfig] = None,
        test_dataloader_conf: Optional[DictConfig] = None,
        fold_num: int = None,
        n_fold: int = None,
    ):
        super().__init__()
        self.train_data_dir = train_data_dir
        self.val_data_dir = val_data_dir
        self.test_data_dir = test_data_dir
        self.df_path = df_path
        self.submit_df_path = submit_df_path
        self.train_dataset_conf = train_dataset_conf or OmegaConf.create()
        self.val_dataset_conf = val_dataset_conf or OmegaConf.create()
        self.test_dataset_conf = test_dataset_conf or OmegaConf.create()
        self.train_dataloader_conf = train_dataloader_conf or OmegaConf.create()
        self.val_dataloader_conf = val_dataloader_conf or OmegaConf.create()
        self.test_dataloader_conf = test_dataloader_conf or OmegaConf.create()
        self.fold_num = fold_num if fold_num is not None else 0
        self.n_fold = n_fold if n_fold is not None else 5

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            df = pd.read_csv(self.df_path)
            skf = StratifiedKFold(n_splits=self.n_fold, shuffle=True)
            df.loc[:, 'fold'] = 0
            for fold_num, (train_index, val_index) in enumerate(skf.split(X=df.index, y=df.label.values)):
                df.loc[df.iloc[val_index].index, 'fold'] = fold_num

            train_df = df[df.fold != self.fold_num].reset_index(drop=True)
            val_df = df[df.fold == self.fold_num].reset_index(drop=True)
            self.train = CassavaDataset(self.train_data_dir, train_df, train=True,
                                        **self.train_dataset_conf)
            self.val = CassavaDataset(self.val_data_dir, val_df, train=False,
                                      **self.val_dataset_conf)

        if stage == "test" or stage is None:
            sub = pd.read_csv(self.submit_df_path)
            self.test = CassavaDataset(self.test_data_dir, sub, train=False,
                                       **self.test_dataset_conf)

    def train_dataloader(self):
        return DataLoader(self.train, **self.train_dataloader_conf)

    def val_dataloader(self):
        return DataLoader(self.val, **self.val_dataloader_conf)

    def test_dataloader(self):
        return DataLoader(self.test, **self.test_dataloader_conf)