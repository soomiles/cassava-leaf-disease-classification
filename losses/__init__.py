import torch.nn as nn
from losses.bi_tempered_logistic_loss import bi_tempered_logistic_loss
from losses.factory import CrossEntropyOneHot, LabelSmoothing, F1_Loss
from functools import partial
from typing import Tuple, List, Optional
from omegaconf import DictConfig, OmegaConf


def create_loss(name: str,
                loss_conf: Optional[DictConfig] = None):
    if name == 'bi_tempered_logistic_loss':
        return partial(bi_tempered_logistic_loss, **loss_conf)
    elif name == 'cross_entropy_loss':
        return nn.CrossEntropyLoss(**loss_conf)
    elif name == 'cross_entropy_onehot':
        return CrossEntropyOneHot()
    elif name == 'label_smoothing_onehot':
        return LabelSmoothing(**loss_conf)
    elif name == 'f1_loss_onehot':
        return F1_Loss()
    else:
        ValueError(f'loss - {name} is invalid.')
