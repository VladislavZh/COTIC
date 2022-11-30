from typing import Any, List, Optional
from collections.abc import Iterable

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


class BaseEventModule(LightningModule):
    """
    Base event sequence lightning module
    """

    def __init__(
        self,
        net: torch.nn.Module,
        metrics: MetricsCore,
        optimizer: dict,
        scheduler: Optional[dict] = None,
        head_start: Optional[int] = None
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.net = net
        self.train_metrics = metrics
        self.val_metrics = metrics.copy_empty()
        self.test_metrics = metrics.copy_empty()

        self.start_time = time.time()

    def forward(self, batch):
        return self.net(*batch)

    def step(self, batch: Any, stage: str):
        outputs = self.forward(batch)

        if stage == 'train':
            loss = self.train_metrics.compute_loss_and_add_values(self, batch, outputs)
        if stage == 'val':
            loss = self.val_metrics.compute_loss_and_add_values(self, batch, outputs)
        if stage == 'test':
            loss = self.test_metrics.compute_loss_and_add_values(self, batch, outputs)

        return loss, outputs

    def training_step(self, batch: Any, batch_idx: int):
        loss, out = self.step(batch, 'train')

        if type(loss) != torch.Tensor:
            assert len(loss) == 2

            self.log("train/loss", loss[0] + loss[1], on_step=False, on_epoch=True, prog_bar=False)
            print(loss[0], loss[1])

            if self.hparams.head_start is not None:
                if self.current_epoch>=self.hparams.head_start:
                    return {"loss": loss[0] + loss[1]}
            return {"loss": loss[0]}

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)

        return {"loss": loss, "out": out}

    def training_epoch_end(self, outputs: List[Any]):
        ll, return_time_metric, event_type_metric = self.train_metrics.compute_metrics()
        self.train_metrics.clear_values()

        self.log("train/log_likelihood", ll, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/return_time_metric", return_time_metric, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/event_type_metric", event_type_metric, on_step=False, on_epoch=True, prog_bar=True)

    def validation_step(self, batch: Any, batch_idx: int):
        loss, out = self.step(batch, 'val')

        if type(loss) != torch.Tensor:
            assert len(loss) == 2

            self.log("val/loss", loss[0] + loss[1], on_step=False, on_epoch=True, prog_bar=False)
            return {"loss": loss[0] + loss[1]}

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)

        return {"loss": loss, "out": out}

    def validation_epoch_end(self, outputs: List[Any]):
        ll, return_time_metric, event_type_metric = self.val_metrics.compute_metrics()
        self.val_metrics.clear_values()

        self.log("val/log_likelihood", ll, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/return_time_metric", return_time_metric, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/event_type_metric", event_type_metric, on_step=False, on_epoch=True, prog_bar=True)
        self.log("training_time", time.time() - self.start_time, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        loss, out = self.step(batch, 'test')

        if type(loss) != torch.Tensor:
            assert len(loss) == 2

            self.log("test/loss", loss[0] + loss[1], on_step=False, on_epoch=True, prog_bar=False)
            return {"loss": loss[0] + loss[1]}

        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=False)

        return {"loss": loss, "out": out}

    def test_epoch_end(self, outputs: List[Any]):
        ll, return_time_metric, event_type_metric = self.test_metrics.compute_metrics()
        self.test_metrics.clear_values()

        self.log("test/log_likelihood", ll, on_step=False, on_epoch=True)
        self.log("test/return_time_metric", return_time_metric, on_step=False, on_epoch=True)
        self.log("test/event_type_metric", event_type_metric, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = get_optimizer(self.hparams.optimizer['name'], self.net.parameters(), self.hparams.optimizer['params'])
        schedulers = []
        if self.hparams.scheduler is not None:
            params = self.hparams.scheduler
            assert params.step is None or params.milestones is None
            if params.step is not None:
                schedulers = [torch.optim.lr_scheduler.StepLR(
                    optimizer,
                    params.step,
                    gamma=params.gamma)]
            elif params.milestones is not None:
                schedulers = [torch.optim.lr_scheduler.MultiStepLR(
                    optimizer,
                    params.milestones,
                    gamma=params.gamma)]
        return [optimizer],schedulers
