import torch.nn as nn
from losses.bi_tempered_logistic_loss import bi_tempered_logistic_loss,\
    noise_bi_tempered_logistic_loss
from losses.factory import CrossEntropyOneHot, LabelSmoothingOneHot,\
    F1_Loss, NoiseCrossEntropyOneHot, NoiseLabelSmoothingOneHot
from losses.taylor_cross_entropy import TaylorCrossEntropyLoss
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
    elif name == 'noise_cross_entropy_onehot':
        return NoiseCrossEntropyOneHot()
    elif name == 'noise_label_smoothing_onehot':
        return NoiseLabelSmoothingOneHot()
    elif name == 'noise_bi_tempered_logistic_loss':
        return partial(noise_bi_tempered_logistic_loss, **loss_conf)
    elif name == 'label_smoothing_onehot':
        return LabelSmoothingOneHot(**loss_conf)
    elif name == 'f1_loss_onehot':
        return F1_Loss()
    elif name == 'taylor_cross_entropy_onehot':
        return TaylorCrossEntropyLoss(**loss_conf)
    else:
        ValueError(f'loss - {name} is invalid.')
