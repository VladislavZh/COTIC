from typing import Any, List

import torch
from pytorch_lightning import LightningModule

from src.utils.metrics import MetricsCore


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
        optimizer: dict
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.net = net
        self.metrics = metrics

    def forward(self, batch):
        return self.net(*batch)

    def step(self, batch: Any):
        ouputs = self.forward(batch)
        loss = self.metrics.compute_loss_and_add_values(self, batch, outputs)
        return loss

    def training_step(self, batch: Any, batch_idx: int):
        loss = self.step(batch)
        
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        
        return {"loss": loss}

    def training_epoch_end(self, outputs: List[Any]):
        ll, return_time_metric, event_type_metric = self.metrics.compute_metrics()
        self.metrics.clear_values()
        
        self.log("train/log_likelihood", ll, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/return_time_metric", return_time_metric, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/event_type_metric", event_type_metric, on_step=False, on_epoch=True, prog_bar=True)

    def validation_step(self, batch: Any, batch_idx: int):
        loss = self.step(batch)
        
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        
        return {"loss": loss}

    def validation_epoch_end(self, outputs: List[Any]):
        ll, return_time_metric, event_type_metric = self.metrics.compute_metrics()
        self.metrics.clear_values()
        
        self.log("val/log_likelihood", ll, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/return_time_metric", return_time_metric, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/event_type_metric", event_type_metric, on_step=False, on_epoch=True, prog_bar=True)


    def test_step(self, batch: Any, batch_idx: int):
        loss = self.step(batch)
        
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        
        return {"loss": loss}

    def test_epoch_end(self, outputs: List[Any]):
        ll, return_time_metric, event_type_metric = self.metrics.compute_metrics()
        self.metrics.clear_values()
        
        self.log("test/log_likelihood", ll, on_step=False, on_epoch=True)
        self.log("test/return_time_metric", return_time_metric, on_step=False, on_epoch=True)
        self.log("test/event_type_metric", event_type_metric, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return get_optimizer(self.hparams.optimizer['name'], self.net.parameters(), self.hparams.optimizer['params'])
