import torch
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.metrics import Accuracy
import timm

from losses.bi_tempered_logistic_loss import bi_tempered_logistic_loss
from functools import partial


class LitTrainer(pl.LightningModule):
    def __init__(self, train_config):
        super(LitTrainer, self).__init__()
        # self.save_hyperparameters()
        self.train_config = train_config
        self.model = timm.create_model(**train_config.network)
        self.criterion = partial(bi_tempered_logistic_loss, **train_config.loss)
        self.evaluator = Accuracy()

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        train_loss = self.criterion(y_hat, y)
        self.log('train_loss', train_loss)

        y_hat = torch.nn.functional.softmax(y_hat, dim=1).argmax(dim=1)
        train_score = self.evaluator(y, y_hat)
        self.log('train_score', train_score)
        return train_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        valid_loss = self.criterion(y_hat, y)
        self.log('valid_loss', valid_loss)

        y_hat = torch.nn.functional.softmax(y_hat, dim=1).argmax(dim=1)
        valid_score = self.evaluator(y, y_hat)
        self.log('valid_score', valid_score, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.train_config.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'valid_score'
        }