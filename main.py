import os
import gc
import numpy as np
import pandas as pd
import logging
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import instantiate

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.utilities.seed import seed_everything
from network.LitModule import LitTrainer

import warnings
warnings.filterwarnings('ignore')

# A logger for this file
logger = logging.getLogger(__name__)

@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, cfg.device_list))
    seed_everything(cfg.seed)
    logger.info(f"Training with the following config:\n{OmegaConf.to_yaml(cfg)}")

    scores, ckpt_paths = [], []
    for fold_num in range(cfg.train.n_fold_iter):
        data = instantiate(cfg.dataset, fold_num=fold_num, n_fold=cfg.train.n_fold)

        tb_logger = TensorBoardLogger(save_dir=os.getcwd(),
                                      version=f'fold{fold_num}')
        checkpoint_callback = ModelCheckpoint(dirpath=tb_logger.log_dir,
                                              filename="{epoch:02d}-{valid_score:.4f}",
                                              monitor='valid_score', mode='max', verbose=False)
        early_stop_callback = EarlyStopping(monitor='valid_score', mode='max',
                                            patience=100, verbose=False)
        model = LitTrainer(cfg)
        trainer = pl.Trainer(gpus=len(cfg.device_list), max_epochs=cfg.train.n_epochs,
                             progress_bar_refresh_rate=1,
                             logger=tb_logger, callbacks=[early_stop_callback, checkpoint_callback])
        trainer.fit(model, data)
        ckpt_paths.append(checkpoint_callback.best_model_path)
        scores.append(float(checkpoint_callback.best_model_score))
        logger.info(f'fold {fold_num} - {checkpoint_callback.best_model_score:.4f}')
    logger.info(f'score: {np.mean(scores):.4f} ({os.getcwd()})')
    del data, model, trainer
    gc.collect()
    torch.cuda.empty_cache()

    sub = pd.read_csv(cfg.dataset.submit_df_path)
    data = instantiate(cfg.dataset)
    infer = pl.Trainer(gpus=len(cfg.device_list), accelerator='dp')
    preds = []
    for path in ckpt_paths:
        model = LitTrainer.load_from_checkpoint(path, train_config=cfg).eval()
        pred = infer.test(model, datamodule=data)[0]
        preds.append(pred['prob'])
    sub['label'] = sub['label'].astype(object)
    sub['label'] = np.mean(preds, axis=0).tolist()
    sub.to_csv(os.path.join(os.getcwd(), 'output.csv'), index=False)

if __name__ == "__main__":
    main()