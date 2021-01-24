import os
from glob import glob
from omegaconf import OmegaConf
from collections import OrderedDict
import torch
import torch.nn.functional as F
from timm.optim import create_optimizer
from torch.optim.lr_scheduler import *
from warmup_scheduler import GradualWarmupScheduler

import pytorch_lightning as pl
from pytorch_lightning.core.decorators import auto_move_data
from pytorch_lightning.metrics import Accuracy
from pytorch_lightning.utilities.cloud_io import load as pl_load
import timm
from torch.utils.data import DataLoader

from dataset.transforms import get_transforms
from losses import create_loss


class LitTrainer(pl.LightningModule):
    def __init__(self, train_config):
        super(LitTrainer, self).__init__()
        self.save_hyperparameters()
        self.train_config = train_config
        self.model = timm.create_model(**train_config.network)
        self.criterion = create_loss(train_config.loss.name, train_config.loss.args)
        self.evaluator = Accuracy()

    @auto_move_data
    def forward(self, x):
        x = self.model(x)
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

    def validation_epoch_end(self, outputs):
        if self.train_config.train.do_noise and \
                (self.current_epoch + 1) % self.train_config.train.noise_params.period_epoch == 0:
            self.ensemble_prediction()

    def ensemble_prediction(self):
        transforms = get_transforms(img_size=(self.val_dataloader().dataset.h,
                                              self.val_dataloader().dataset.w))
        batch_size = self.train_config.batch_size

        self.train_dataloader().dataset.transforms = transforms['val']
        for ds in [self.train_dataloader().dataset, self.val_dataloader().dataset]:
            dl = DataLoader(ds, **self.train_config.dataset.val_dataloader_conf)
            for index, batch in enumerate(dl):
                dl_idx = index * batch_size
                x, y = batch
                with torch.no_grad():
                    y_hat = torch.nn.functional.softmax(self(x), dim=1).cpu()
                y_hat = torch.nn.functional.softmax(
                    self.train_config.train.noise_params.alpha * y_hat + torch.from_numpy(
                        (1 - self.train_config.train.noise_params.alpha) * ds.labels_copy[dl_idx: dl_idx + batch_size]),
                    dim=1)
                ds.labels_copy[dl_idx: dl_idx+batch_size] = y_hat.numpy()
        self.train_dataloader().dataset.transforms = transforms['train']

        if self.current_epoch >= self.train_config.train.noise_params.thr_epochs:
            self.train_dataloader().dataset.labels = self.train_dataloader().dataset.labels_copy.copy()
            # self.val_dataloader().dataset.labels = self.val_dataloader().dataset.labels_copy.copy()

    def test_step(self, batch, batch_idx):
        score = torch.nn.functional.softmax(self(batch), dim=1)
        score2 = torch.nn.functional.softmax(self(torch.flip(batch, [-1])), dim=1)
        score3 = torch.nn.functional.softmax(self(torch.flip(batch, [-2])), dim=1)

        out = (score + score2 + score3) / 3.0
        return {"pred": out}

    def test_epoch_end(self, output_results):
        all_outputs = torch.cat([out["pred"] for out in output_results], dim=0)
        all_outputs = all_outputs.cpu().numpy()
        return {'prob': all_outputs}

    def configure_optimizers(self):
        optimizer = create_optimizer(self.train_config.optimizer, self)
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


class DistilledTrainer(LitTrainer):
    def __init__(self, train_config, teacher_dir, fold_num):
        super().__init__(train_config)
        train_config.network.num_classes = 10
        self.model = timm.create_model(**train_config.network)

        teacher_path = glob(os.path.join(teacher_dir, f'checkpoints/*fold{fold_num}*.ckpt'))[0]
        self._teacher_model = self._load_teacher_network(teacher_path)
        self.teacher_criterion = create_loss(train_config.train.distillation_params.loss.name,
                                             train_config.train.distillation_params.loss.args)

    @auto_move_data
    def forward(self, x):
        x = self.model(x)
        return x[:, :5], x[:, 5:]

    def training_step(self, batch, batch_idx):
        x, y = batch
        with torch.no_grad():
            y_teacher = self._teacher_model(x)
        y_hat1, y_hat2 = self(x)
        train_loss1 = self.criterion(y_hat1, y)
        train_loss2 = self.teacher_criterion(y_hat2, y_teacher)
        loss = (train_loss1 + train_loss2) / 2
        self.log('train_loss1', train_loss1)
        self.log('train_loss2', train_loss2)
        self.log('train_loss', loss)

        y = y.argmax(dim=1) if len(y.shape) > 1 else y
        y_hat1, y_hat2, y_teacher = y_hat1.argmax(dim=1), y_hat2.argmax(dim=1), y_teacher.argmax(dim=1)
        train_score1 = self.evaluator(y_hat1, y)
        train_score2 = self.evaluator(y_hat2, y_teacher)
        score = (train_score1 + train_score2) / 2
        self.log('train_score1', train_score1)
        self.log('train_score2', train_score2)
        self.log('train_score', score)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat1, y_hat2 = self(x)
        y_hat = (y_hat1 + y_hat2) / 2
        valid_loss = self.criterion(y_hat, y)
        self.log('valid_loss', valid_loss)

        y = y.argmax(dim=1) if len(y.shape) > 1 else y
        y_hat1, y_hat2, y_hat = y_hat1.argmax(dim=1), y_hat2.argmax(dim=1), y_hat.argmax(dim=1)
        valid_score1 = self.evaluator(y_hat1, y)
        valid_score2 = self.evaluator(y_hat2, y)
        valid_score = self.evaluator(y_hat, y)
        self.log('valid_score1', valid_score1, on_epoch=True)
        self.log('valid_score2', valid_score2, on_epoch=True)
        self.log('valid_score', valid_score, on_epoch=True, prog_bar=True)

    def test_micro_step(self, batch):
        y_hat1, y_hat2 = self(batch)
        y_hat = (y_hat1 + y_hat2) / 2
        return y_hat

    def test_step(self, batch, batch_idx):
        score = torch.nn.functional.softmax(self.test_micro_step(batch), dim=1)
        score2 = torch.nn.functional.softmax(self.test_micro_step(torch.flip(batch, [-1])), dim=1)
        score3 = torch.nn.functional.softmax(self.test_micro_step(torch.flip(batch, [-2])), dim=1)

        out = (score + score2 + score3) / 3.0
        return {"pred": out}

    def _load_teacher_network(self, path):
        cfg_path = f"{'/'.join(path.split('/')[:-2])}/.hydra/config.yaml"
        teacher_cfg = OmegaConf.load(cfg_path)
        teacher_cfg.network.pretrained = False

        _teacher_model = timm.create_model(**teacher_cfg.network)
        _teacher_model = timm.create_model(**teacher_cfg.network)
        _teacher_model.load_state_dict(OrderedDict((k.replace('model.', '') if 'model.' in k else k, v) for k, v in
                                                        pl_load(path, map_location='cpu')['state_dict'].items()))
        _teacher_model.eval()
        for param in _teacher_model.parameters():
            param.requires_grad = False
        return _teacher_model
