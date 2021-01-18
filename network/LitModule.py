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
        self.save_hyperparameters()
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
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.train_config.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'valid_score'
        }
