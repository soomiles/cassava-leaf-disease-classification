import os
import gc
from glob import glob
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

    # Training
    for fold_num in range(cfg.train.n_fold_iter):
        data = instantiate(cfg.dataset, fold_num=fold_num, n_fold=cfg.train.n_fold)

        tb_logger = TensorBoardLogger(save_dir=os.getcwd(),
                                      version=f'fold{fold_num}')
        checkpoint_callback = ModelCheckpoint(dirpath=tb_logger.log_dir,
                                              filename="{epoch:02d}-{valid_score:.4f}",
                                              monitor='valid_score', mode='max', verbose=False)
        early_stop_callback = EarlyStopping(monitor='valid_score', mode='max',
                                            patience=5, verbose=False)
        model = LitTrainer(cfg)
        trainer = pl.Trainer(gpus=len(cfg.device_list), max_epochs=cfg.train.n_epochs,
                             progress_bar_refresh_rate=1,
                             logger=tb_logger, callbacks=[early_stop_callback, checkpoint_callback])
        trainer.fit(model, data)
    del data, model, trainer
    gc.collect()
    torch.cuda.empty_cache()

    # Move Checkpoints
    os.makedirs(os.path.join(os.getcwd(), 'checkpoints'), exist_ok=True)
    ckpt_paths, scores = [], []
    for ckpt_path in glob(os.path.join(os.getcwd(), 'default/**/*.ckpt')):
        if 'tmp_end' in ckpt_path:
            continue
        fold = os.path.basename(os.path.dirname(ckpt_path))
        ckpt_name = os.path.basename(ckpt_path)
        new_ckpt_path = os.path.join(os.getcwd(), 'checkpoints',
                                     f'{"-".join(os.getcwd().split("/")[-2:])}-{fold}-{ckpt_name}')
        os.rename(ckpt_path, new_ckpt_path)
        ckpt_paths.append(new_ckpt_path)
        fold_score = float(os.path.basename(new_ckpt_path).split('valid_score=')[1][:6])
        scores.append(fold_score)
        logger.info(f'Fold {fold} - {fold_score:.4f}')
    logger.info(f'score: {np.mean(scores):.4f} ({os.getcwd()})')
    summary_df = pd.read_csv('/workspace/logs/cassava-leaf-disease-classification/summary.csv')
    summary_df = summary_df.append(pd.Series(["/".join(os.getcwd().split("/")[-3:]), np.mean(scores)],
                                             index=['path', 'score']), ignore_index=True)
    summary_df.to_csv('/workspace/logs/cassava-leaf-disease-classification/summary.csv', index=False)

    # Inference
    if cfg.train.run_test:
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