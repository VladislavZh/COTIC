from typing import Any, List

import torch
from pytorch_lightning import LightningModule

from src.utils.metrics import MetricsCore

import time


def get_optimizer(name, model_params, params):
    optimizers = {
        "adadelta": torch.optim.Adadelta,
        "adagrad": torch.optim.Adagrad,
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "sparseadam": torch.optim.SparseAdam,
        "adamax": torch.optim.Adamax,
        "asgd": torch.optim.ASGD,
        "lbfgs": torch.optim.LBFGS,
        "nadam": torch.optim.NAdam,
        "radam": torch.optim.RAdam,
        "rmsprop": torch.optim.RMSprop,
        "rprop": torch.optim.Rprop,
        "sgd": torch.optim.SGD
    }
    return optimizers[name](params=model_params, **params)


class ExtrHeadEventModule(LightningModule):
    """
    Base event sequence lightning module
    """

    def __init__(
        self,
        net: torch.nn.Module,
        head: torch.nn.Module,
        metrics: MetricsCore,
        optimizers: dict,
        enc_output_only_head: bool = True
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.net = net
        self.head = head
        self.enc_output_only_head = enc_output_only_head
        self.train_metrics = metrics
        self.val_metrics = metrics.copy_empty()
        self.test_metrics = metrics.copy_empty()

        self.start_time = None

    def forward(self, batch):
        net_output = self.net(*batch)
        if self.enc_output_only_head:
            head_output = self.head(batch, net_output.detach())
        else:
            head_output = self.head(*batch, net_output.detach(), self)
        return net_output, head_output

    def step(self, batch: Any, stage: str):
        outputs = self.forward(batch)

        if stage == 'train':
            loss1, loss2 = self.train_metrics.compute_loss_and_add_values(self, batch, outputs)
        if stage == 'val':
            loss1, loss2 = self.val_metrics.compute_loss_and_add_values(self, batch, outputs)
        if stage == 'test':
            loss1, loss2 = self.test_metrics.compute_loss_and_add_values(self, batch, outputs)

        return loss1, loss2

    def on_train_start(self):
        self.start_time = time.time()

    def training_step(self, batch: Any, batch_idx: int):#, optimizer_idx):
        loss1, loss2 = self.step(batch, 'train')

        self.log("train/loss", loss1 + loss2, on_step=False, on_epoch=True, prog_bar=False)

        print(loss1, loss2)

        return {"loss": loss1 + loss2}

        # if optimizer_idx == 0:
        #     return {"loss": loss1}
        # else:
        #     return {"loss": loss2}

    def training_epoch_end(self, outputs: List[Any]):
        ll, return_time_metric, event_type_metric = self.train_metrics.compute_metrics()
        self.train_metrics.clear_values()

        self.log("train/log_likelihood", ll, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/return_time_metric", return_time_metric, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/event_type_metric", event_type_metric, on_step=False, on_epoch=True, prog_bar=True)
        self.log("training_time", time.time() - self.start_time, on_step=False, on_epoch=True, prog_bar=True)

    def validation_step(self, batch: Any, batch_idx: int):
        loss1, loss2 = self.step(batch, 'val')

        self.log("val/loss", loss1 + loss2, on_step=False, on_epoch=True, prog_bar=False)

        return {"loss": loss1 + loss2}

    def validation_epoch_end(self, outputs: List[Any]):
        ll, return_time_metric, event_type_metric = self.val_metrics.compute_metrics()
        self.val_metrics.clear_values()

        self.log("val/log_likelihood", ll, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/return_time_metric", return_time_metric, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/event_type_metric", event_type_metric, on_step=False, on_epoch=True, prog_bar=True)


    def test_step(self, batch: Any, batch_idx: int):
        loss1, loss2 = self.step(batch, 'test')

        self.log("test/loss", loss1 + loss2, on_step=False, on_epoch=True)

        return {"loss": loss1 + loss2}

    def test_epoch_end(self, outputs: List[Any]):
        ll, return_time_metric, event_type_metric = self.test_metrics.compute_metrics()
        self.test_metrics.clear_values()

        self.log("test/log_likelihood", ll, on_step=False, on_epoch=True)
        self.log("test/return_time_metric", return_time_metric, on_step=False, on_epoch=True)
        self.log("test/event_type_metric", event_type_metric, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = get_optimizer(self.hparams.optimizers[0]['name'], torch.nn.ModuleList([self.net, self.head]).parameters(), self.hparams.optimizers[0]['params'])
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, self.hparams.optimizers[0]['milestones'], gamma=self.hparams.optimizers[0]['gamma'], verbose=True)
        return [optimizer], [scheduler]
        # optimizer1 = get_optimizer(self.hparams.optimizers[0]['name'], self.net.parameters(), self.hparams.optimizers[0]['params'])
        # optimizer2 = get_optimizer(self.hparams.optimizers[1]['name'], self.head.parameters(), self.hparams.optimizers[1]['params'])
        # return [optimizer1, optimizer2], []
