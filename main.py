import os
import pandas as pd
import logging
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import instantiate
from pytorch_lightning.utilities.seed import seed_everything

from sklearn.model_selection import StratifiedKFold

from dataset.transforms import get_transforms
from dataset.cassava import CassavaDataset

# A logger for this file
logger = logging.getLogger(__name__)

@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    logger.info(f"Training with the following config:\n{OmegaConf.to_yaml(cfg)}")

    data = instantiate(cfg.dataset, fold_num=0)


if __name__ == "__main__":
    main()