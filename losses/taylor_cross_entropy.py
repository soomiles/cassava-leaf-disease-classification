# code is taken from https://www.kaggle.com/yerramvarun/cassava-taylorce-loss-label-smoothing-combo
# implementations reference - https://github.com/CoinCheung/pytorch-loss/blob/master/pytorch_loss/taylor_softmax.py
# paper - https://www.ijcai.org/Proceedings/2020/0305.pdf
import torch
import torch.nn as nn
from losses.factory import LabelSmoothingLoss, LabelSmoothingOneHot


class TaylorSoftmax(nn.Module):

    def __init__(self, dim=1, n=2):
        super(TaylorSoftmax, self).__init__()
        assert n % 2 == 0
        self.dim = dim
        self.n = n

    def forward(self, x):
        fn = torch.ones_like(x)
        denor = 1.
        for i in range(1, self.n + 1):
            denor *= i
            fn = fn + x.pow(i) / denor
        out = fn / fn.sum(dim=self.dim, keepdims=True)
        return out


class TaylorCrossEntropyLoss(nn.Module):

    def __init__(self, num_classes,
                 n=2, ignore_index=-1, reduction='mean', smoothing=0.2, onehot=True):
        super(TaylorCrossEntropyLoss, self).__init__()
        assert n % 2 == 0
        self.taylor_softmax = TaylorSoftmax(dim=1, n=n)
        self.reduction = reduction
        self.ignore_index = ignore_index
        if onehot:
            self.lab_smooth = LabelSmoothingOneHot(smoothing=smoothing, log_softmax=False)
        else:
            self.lab_smooth = LabelSmoothingLoss(num_classes, smoothing=smoothing, log_softmax=False)

    def forward(self, logits, labels):
        log_probs = self.taylor_softmax(logits).log()
        # loss = F.nll_loss(log_probs, labels, reduction=self.reduction,
        #        ignore_index=self.ignore_index)
        loss = self.lab_smooth(log_probs, labels)
        return loss
