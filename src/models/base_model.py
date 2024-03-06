from typing import Any, Union, Optional, Callable
import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.core.optimizer import LightningOptimizer
from torch.optim import Optimizer

from src.models.components.cotic.head.downstream_head import DownstreamHead
from src.models.components.cotic.head.intensity_head import IntensityHead
from src.models.components.cotic.head.joined_head import JoinedHead


class BaseEventModule(LightningModule):
    """
    Base event sequence lightning module for neural network models.
    """

    def __init__(
            self,
            net: torch.nn.Module,
            joined_head: JoinedHead,
            optimizer: torch.optim.Optimizer,
            init_lr: float,
            scheduler: torch.optim.lr_scheduler,
            warmup_steps: int,
            scheduler_monitoring_params: dict
    ) -> None:
        """
        Initialize BaseEventModule.

        Args:
        - net (torch.nn.Module): The neural network model.
        - joined_head (JoinedHead): Head object for intensity predictions and downstream predictions.
        - optimizer (torch.optim.Optimizer): Optimizer for model training.
        - scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
        - scheduler_monitoring_params (dict): Parameters for monitoring the scheduler.
        """
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.net = net
        self.joined_head = joined_head

    def forward(self, batch: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        """
        Perform forward pass through the neural network.

        Args:
        - batch: Input data_utils batch.

        Returns:
        - Output from the neural network.
        """
        return self.net(*batch)

    def step(self, batch: Any, stage: str) -> tuple[torch.Tensor, float, float, float]:
        """
        Perform a training/validation/testing step.

        Args:
        - batch: Input data_utils batch.
        - stage (str): Stage of operation (train/val/test).

        Returns:
        - Tuple containing loss value, intensity predictions, and downstream predictions.
        """
        times, events = batch
        non_pad_mask = events.ne(0)

        embeddings = self.forward(batch)
        intensity_prediction, downstream_predictions = self.joined_head(
            times,
            events,
            embeddings,
            non_pad_mask,
            self.trainer.datamodule.normalizer,
            stage
        )

        return (
            intensity_prediction.loss + downstream_predictions.loss
            if downstream_predictions.loss is not None
            else intensity_prediction.loss,
            intensity_prediction.loss,
            downstream_predictions.metrics["return_time_mae"],
            downstream_predictions.metrics["event_type_accuracy"]
        )

    def log_downstream(
            self,
            return_time_mae: float,
            event_type_accuracy: float,
            stage: str
    ) -> None:
        """
        Log downstream predictions' metrics for a given stage.

        Args:
        - downstream_predictions (DownstreamPredictions): Predictions from downstream.
        - stage (str): Stage of operation (train/val/test).
        """
        self.log(f"{stage}/return_time_mae", return_time_mae, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{stage}/event_type_accuracy", event_type_accuracy, on_step=False, on_epoch=True, prog_bar=True)

    def run_evaluation(self, batch: Any, stage: str) -> torch.Tensor:
        """
        Evaluation step for training, validation, or test.

        Args:
        - batch: Input data_utils batch.
        - stage: Stage of operation (train/val/test).

        Returns:
        - torch.Tensor: Loss value obtained during evaluation.
        """
        loss, negative_log_likelihood, return_time_mae, event_type_accuracy = self.step(batch, stage)
        self.log(f"{stage}/log_likelihood", -negative_log_likelihood, on_step=stage == "train",
                 on_epoch=stage != "train", prog_bar=False)
        self.log_downstream(return_time_mae, event_type_accuracy, stage)
        self.log(f"{stage}/loss", loss, on_step=stage == "train", on_epoch=stage != "train", prog_bar=False)

        return loss

    def training_step(self, batch: Any, batch_idx: int):
        """
        Training step.

        Args:
        - batch: Input data_utils batch.
        - batch_idx: Index of the current batch.

        Returns:
        - Dictionary with the loss value.
        """
        loss = self.run_evaluation(batch, "train")

        cur_lr = None
        optimizers = self.trainer.optimizers
        for optimizer in optimizers:
            for param_group in optimizer.param_groups:
                cur_lr = param_group["lr"]
                break
            break

        self.log("lr", cur_lr, on_step=True, on_epoch=False, prog_bar=True)

        return {"loss": loss}

    def validation_step(self, batch: Any, batch_idx: int):
        """
        Validation step.

        Args:
        - batch: Input data_utils batch.
        - batch_idx: Index of the current batch.
        """
        self.run_evaluation(batch, "val")

    def test_step(self, batch: Any, batch_idx: int):
        """
        Test step.

        Args:
        - batch: Input data_utils batch.
        - batch_idx: Index of the current batch.
        """
        self.run_evaluation(batch, "test")

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: Union[Optimizer, LightningOptimizer],
        optimizer_idx: int = 0,
        optimizer_closure: Optional[Callable[[], Any]] = None,
        on_tpu: bool = False,
        using_native_amp: bool = False,
        using_lbfgs: bool = False,
    ) -> None:
        # manually warm up lr without a scheduler
        if self.trainer.global_step < self.hparams.warmup_steps:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / self.hparams.warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.hparams.init_lr

        # update params
        if optimizer_closure is not None:
            optimizer_closure()
        optimizer.step()

    def configure_optimizers(self):
        """
        Configure optimizers for training.

        Returns:
        - Dictionary containing optimizer and optional learning rate scheduler.
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10, min_lr=1e-6)
            scheduler_config = dict(self.hparams.scheduler_monitoring_params)
            scheduler_config["scheduler"] = scheduler

            return [optimizer], [scheduler_config]
        return {"optimizer": optimizer}
