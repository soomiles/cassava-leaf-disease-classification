import os
import gc
from glob import glob
import numpy as np
import pandas as pd
import logging
from omegaconf import DictConfig, OmegaConf
from collections import OrderedDict
import hydra
from hydra.utils import instantiate
from sklearn.model_selection import StratifiedKFold

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.utilities.seed import seed_everything

from network.LitModule import LitTrainer, DistilledTrainer, LitTester
from scripts.utils import get_state_dict_from_checkpoint

import warnings
warnings.filterwarnings('ignore')

# A logger for this file
logger = logging.getLogger(__name__)

@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, cfg.device_list))
    seed_everything(cfg.seed)
    logger.info(f"Training with the following config:\n{OmegaConf.to_yaml(cfg)}")
    logger.info(f"Current Path: {'/'.join(os.getcwd().split('/')[-2:])}")
    is_multi_gpu = 'tmp_end' if len(cfg.device_list) > 1 else ''

    # Training & Inference
    df = pd.read_csv(cfg.df_path)
    if (cfg.seed != 42) or 'fold' not in df.columns:
        skf = StratifiedKFold(n_splits=cfg.train.n_fold, shuffle=True)
        df.loc[:, 'fold'] = 0
        for fold_num, (train_index, val_index) in enumerate(skf.split(X=df.index, y=df.label.values)):
            df.loc[df.iloc[val_index].index, 'fold'] = fold_num

    os.makedirs(os.path.join(os.getcwd(), 'checkpoints'), exist_ok=True)
    scores = []
    oof_dict = {'image_id': [], 'label': [], 'fold': []}
    for fold_num in cfg.train.use_fold:
        tb_logger = TensorBoardLogger(save_dir=os.getcwd(),
                                      version=f'fold{fold_num}')
        checkpoint_callback = ModelCheckpoint(dirpath=tb_logger.log_dir,
                                              filename="{epoch:02d}-{valid_score:.4f}",
                                              monitor='valid_score', mode='max', verbose=False)
        early_stop_callback = EarlyStopping(monitor='valid_score', mode='max',
                                            patience=cfg.train.n_epochs//5, verbose=False)
        if cfg.train.do_distillation:
            model = DistilledTrainer(cfg, fold_num=fold_num)
        else:
            model = LitTrainer(cfg, fold_num=fold_num)

        data = instantiate(cfg.dataset, df=df, fold_num=fold_num)
        trainer = pl.Trainer(gpus=len(cfg.device_list),
                             accumulate_grad_batches={cfg.train.total_epoch+1: 2},
                             max_epochs=cfg.train.n_epochs,
                             progress_bar_refresh_rate=1,
                             logger=tb_logger, callbacks=[checkpoint_callback, early_stop_callback])
        trainer.fit(model, data)

        # Move Checkpoint
        ckpt_path = glob(os.path.join(os.getcwd(),
                                      f'default/fold{fold_num}/*{is_multi_gpu}.ckpt'))[0]
        ckpt_name = os.path.basename(ckpt_path)
        new_ckpt_path = os.path.join(os.getcwd(), 'checkpoints',
                                     f'{"-".join(os.getcwd().split("/")[-2:])}-fold{fold_num}-{ckpt_name}')
        os.rename(ckpt_path, new_ckpt_path)
        score = float(os.path.basename(new_ckpt_path).split('valid_score=')[1][:6])
        scores.append(score)
        logger.info(f'Fold {fold_num} - {score:.4f}')
        del trainer, model

        # Inference Hold-out sets
        if cfg.train.run_test:
            state_dict = get_state_dict_from_checkpoint(os.getcwd(), fold_num)
            model = LitTester(cfg.network, state_dict)
            infer = pl.Trainer(gpus=len(cfg.device_list), accelerator='dp')
            pred = infer.test(model, datamodule=data, verbose=False)[0]
            oof_dict['image_id'].extend(df[df.fold == fold_num].image_id.values.tolist())
            oof_dict['label'].extend(pred['prob'].tolist())
            oof_dict['fold'].extend([fold_num] * pred['prob'].shape[0])
            del infer, model
        del data
        gc.collect()
        torch.cuda.empty_cache()

    logger.info(f"score: {np.mean(scores):.4f} ({'/'.join(os.getcwd().split('/')[-2:])})")
    if cfg.train.run_test:
        pd.DataFrame(oof_dict).to_csv(os.path.join(os.getcwd(), 'oof.csv'), index=False)


if __name__ == "__main__":
    main()