import torch
import torch.nn.functional as F
from torch.optim import *
from timm.optim import create_optimizer
from torch.optim.lr_scheduler import *
from warmup_scheduler import GradualWarmupScheduler
from ast import literal_eval

import pytorch_lightning as pl
from pytorch_lightning.core.decorators import auto_move_data
from pytorch_lightning.metrics import Accuracy
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
                    x = torch.nn.functional.softmax(self(x), dim=1).cpu()
                ds.labels_copy[dl_idx: dl_idx+batch_size] = \
                    self.train_config.train.noise_params.alpha * x + \
                    (1 - self.train_config.train.noise_params.alpha) * ds.labels_copy[dl_idx: dl_idx+batch_size]
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
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
        # scheduler = StepLR(optimizer, step_size=self.train_config.train.step_size, gamma=0.5)
        scheduler = GradualWarmupScheduler(optimizer, multiplier=1,
                                           total_epoch=self.train_config.train.total_epoch,
                                           after_scheduler=scheduler)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
        }
