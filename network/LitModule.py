import torch
import torch.nn.functional as F
from timm.optim import create_optimizer
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import *
from warmup_scheduler import GradualWarmupScheduler
from typing import Optional, Callable

import pytorch_lightning as pl
from pytorch_lightning.core.decorators import auto_move_data
from pytorch_lightning.metrics import Accuracy

from losses import create_loss
from scripts.utils import get_state_dict_from_checkpoint, freeze_top_layers
from scripts.sam import SAM
from network.factory import create_model


class LitTrainer(pl.LightningModule):
    def __init__(self, train_config, fold_num):
        super(LitTrainer, self).__init__()
        self.save_hyperparameters()
        self.train_config = train_config
        self.model = create_model(**train_config.network)
        self.criterion = create_loss(train_config.loss.name, train_config.loss.args)
        self.evaluator = Accuracy()

        if self.train_config.train.do_load_ckpt:
            self._load_trained_weight(fold_num, **self.train_config.train.ckpt_params)

    @auto_move_data
    def forward(self, x):
        x = self.model(x)
        if not isinstance(x, torch.Tensor):
            return (x[0] + x[1]) / 2
        else:
            return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        train_loss = self.criterion(y_hat, y)
        self.log('train_loss', train_loss)

        y = y.argmax(dim=1) if len(y.shape) > 1 else y
        y_hat = y_hat.argmax(dim=1)
        train_score = self.evaluator(y_hat, y)
        self.log('train_score', train_score)
        return train_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        valid_loss = self.criterion(y_hat, y)
        self.log('valid_loss', valid_loss)

        y = y.argmax(dim=1) if len(y.shape) > 1 else y
        y_hat = y_hat.argmax(dim=1)
        valid_score = self.evaluator(y_hat, y)
        self.log('valid_score', valid_score, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = create_optimizer(self.train_config.optimizer, self)
        # optimizer = SAM(self.train_config.optimizer, self)
        # optimizer = AdamW(self.parameters(), lr=self.train_config.learning_rate)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=1)
        # scheduler = StepLR(optimizer, step_size=self.train_config.train.step_size, gamma=0.5)
        scheduler = GradualWarmupScheduler(optimizer, multiplier=1,
                                           total_epoch=self.train_config.train.total_epoch,
                                           after_scheduler=scheduler)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
        }

    def _load_trained_weight(self, fold_num, log_path, do_freeze_top_layers):
        state_dict = get_state_dict_from_checkpoint(log_path, fold_num)

        self.model = create_model(**self.train_config.network)
        self.model.load_state_dict(state_dict)
        if do_freeze_top_layers:
            self.model = freeze_top_layers(self.model, self.train_config.network.model_name)
        self.train_config.optimizer.lr /= 5


class DistilledTrainer(LitTrainer):
    def __init__(self, train_config, fold_num):
        super().__init__(train_config, fold_num)
        self.model = create_model(**train_config.network)
        if self.train_config.train.do_load_ckpt:
            self._load_trained_weight(fold_num, **self.train_config.train.ckpt_params)

        self._teacher_model = self._load_teacher_network(train_config.train.distillation_params, fold_num)
        self.teacher_criterion = create_loss(train_config.train.distillation_params.loss.name,
                                             train_config.train.distillation_params.loss.args)

    @auto_move_data
    def forward(self, x):
        x = self.model(x)

        if not isinstance(x, torch.Tensor):
            out = x[0], x[1]
        else:
            out = x # for DeiT.eval()

        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        with torch.no_grad():
            y_teacher = self._teacher_model(x)
            if not isinstance(y_teacher, torch.Tensor):
                y_teacher = (y_teacher[0] + y_teacher[1])/2
            elif y_teacher.shape[1] > 5:
                y_teacher = (y_teacher[:, :5] + y_teacher[:, 5:]) / 2
            y_teacher = F.softmax(y_teacher, dim=-1)

        y_hat = self(x)
        if not isinstance(y_hat, torch.Tensor):
            y_hat1, y_hat2 = y_hat
            train_loss1 = self.criterion(y_hat1, y)
            train_loss2 = self.teacher_criterion(y_hat2, y_teacher)
            loss = (train_loss1 + train_loss2) / 2
        else:
            loss = self.teacher_criterion(y_hat, y_teacher)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        if not isinstance(y_hat, torch.Tensor):
            y_hat = (y_hat[0] + y_hat[1]) / 2
        valid_loss = self.criterion(y_hat, y)
        self.log('valid_loss', valid_loss)

        y = y.argmax(dim=1) if len(y.shape) > 1 else y
        y_hat = y_hat.argmax(dim=1)
        valid_score = self.evaluator(y_hat, y)
        self.log('valid_score', valid_score, on_epoch=True, prog_bar=True)

    def _load_teacher_network(self, distillation_params, fold_num):
        state_dict = get_state_dict_from_checkpoint(distillation_params.dir, fold_num)

        _teacher_model = create_model(model_name=distillation_params.model_name,
                                      pretrained=False,
                                      num_classes=5,
                                      in_chans=3)
        _teacher_model.load_state_dict(state_dict)
        _teacher_model.eval()
        for param in _teacher_model.parameters():
            param.requires_grad = False
        return _teacher_model


class LitTester(pl.LightningModule):
    def __init__(self, network_cfg, state_dict):
        super(LitTester, self).__init__()
        self.model = create_model(**network_cfg)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def forward(self, x):
        x = self.model(x)
        return x

    def test_micro_step(self, x):
        y_hat = self(x)
        if not isinstance(y_hat, torch.Tensor):
            y_hat = (y_hat[0] + y_hat[1]) / 2
        elif y_hat.shape[1] > 5:
            y_hat = (y_hat[:, 5] + y_hat[:, 5:]) / 2
        return y_hat

    def test_step(self, batch, batch_idx):
        x, y = batch
        score = torch.nn.functional.softmax(self.test_micro_step(x), dim=1)
        score2 = torch.nn.functional.softmax(self.test_micro_step(torch.flip(x, [-1])), dim=1)
        score3 = torch.nn.functional.softmax(self.test_micro_step(torch.flip(x, [-2])), dim=1)

        out = (score + score2 + score3) / 3.0
        return {"pred": out}

    def test_epoch_end(self, output_results):
        all_outputs = torch.cat([out["pred"] for out in output_results], dim=0)
        all_outputs = all_outputs.cpu().numpy()
        return {'prob': all_outputs}
