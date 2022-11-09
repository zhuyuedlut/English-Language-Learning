from typing import Any, List

import torch
import torch.nn as nn

from torchmetrics import MeanMetric, MaxMetric
from torchmetrics.classification.accuracy import Accuracy

from pytorch_lightning import LightningModule

from src.models.components.loss.RMSELoss import RMSELoss


class FBModule(LightningModule):
    def __init__(
            self,
            net: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler,
    ):
        super(FBModule, self).__init__()

        self.save_hyperparameters(logger=False, ignore=["net"])

        self.net = net

        self.val_acc = Accuracy()

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

        self.val_loss_best = MaxMetric()

        self.loss = nn.SmoothL1Loss(reduction='mean')

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Any:
        return self.net(input_ids=input_ids, attention_mask=attention_mask)

    def step(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        output = self.forward(input_ids, attention_mask)

        return output

    def training_step(self, batch: Any, batch_idx: int):
        input_ids, attention_mask, labels = batch
        logits = self.step(input_ids=input_ids, attention_mask=attention_mask)
        loss = self.loss(logits, labels)

        self.train_loss(loss)

        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)

        return {"loss": loss, "labels": labels}

    def validation_step(self, batch: Any, batch_idx: int):
        input_ids, attention_mask, labels = batch
        logits= self.step(input_ids=input_ids, attention_mask=attention_mask)
        loss = self.loss(logits, labels)

        self.val_loss(loss)
        # Lightning 会根据on_step和on_epoch来记录metric，如果on_epoch=True logger会在epoch结束的时候自动调用compute
        self.log("val/loss", self.val_loss, on_step=True, on_epoch=True, prog_bar=True)

        return {"loss": loss, "labels": labels}

    def validation_epoch_end(self, outputs: List[Any]):
        loss = self.val_loss.compute()  # get current val acc
        self.val_loss_best(loss)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/loss_best", self.val_loss_best.compute(), prog_bar=True)

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.parameters())
        scheduler = self.hparams.scheduler(
            optimizer=optimizer,
            # temp set 1000 in order to debugger
            num_training_steps=1000
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }