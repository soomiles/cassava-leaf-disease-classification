import os
import pandas as pd
import logging
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import instantiate

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.utilities.seed import seed_everything
from network.LitModule import LitTrainer

from sklearn.model_selection import StratifiedKFold

from dataset.transforms import get_transforms
from dataset.cassava import CassavaDataset

# A logger for this file
logger = logging.getLogger(__name__)

@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, cfg.device_list))
    seed_everything(cfg.seed)
    logger.info(f"Training with the following config:\n{OmegaConf.to_yaml(cfg)}")

    for fold_num in range(cfg.train.n_fold_iter):
        data = instantiate(cfg.dataset, fold_num=0)

        tb_logger = TensorBoardLogger(save_dir=os.getcwd(),
                                      version=f'fold{fold_num}')
        checkpoint_callback = ModelCheckpoint(dirpath=tb_logger.log_dir,
                                              filename="{epoch:02d}-{valid_score:.4f}",
                                              monitor='valid_score', mode='max', verbose=True)
        early_stop_callback = EarlyStopping(monitor='valid_score', mode='max',
                                            patience=100, verbose=True)
        model = LitTrainer(cfg)
        trainer = pl.Trainer(gpus=len(cfg.device_list), max_epochs=cfg.train.n_epochs,
                             progress_bar_refresh_rate=1,
                             logger=tb_logger, callbacks=[early_stop_callback, checkpoint_callback])
        trainer.fit(model, data)


if __name__ == "__main__":
    main()