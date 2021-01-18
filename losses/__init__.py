import torch.nn as nn
from losses.bi_tempered_logistic_loss import bi_tempered_logistic_loss
from functools import partial
from typing import Tuple, List, Optional
from omegaconf import DictConfig, OmegaConf

def create_loss(name: str,
                loss_conf: Optional[DictConfig] = None):
    if name == 'bi_tempered_logistic_loss':
        return partial(bi_tempered_logistic_loss, **loss_conf)
    elif name == 'cross_entropy_loss':
        return nn.CrossEntropyLoss(**loss_conf)
    else:
        ValueError(f'loss - {name} is invalid.')
