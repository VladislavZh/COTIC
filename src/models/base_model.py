from typing import Any
import torch
from pytorch_lightning import LightningModule
from src.utils.head.head_core import JoinedHead
from src.utils.structures import DownstreamPredictions, Predictions
import time


class BaseEventModule(LightningModule):
    """
    Base event sequence lightning module for neural network models.
    """

    def __init__(
            self,
            net: torch.nn.Module,
            joined_head: JoinedHead,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler,
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
        self.start_time = time.time()

    def forward(self, batch: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        """
        Perform forward pass through the neural network.

        Args:
        - batch: Input data batch.

        Returns:
        - Output from the neural network.
        """
        return self.net(*batch)

    def step(self, batch: Any, stage: str) -> tuple[torch.Tensor, Predictions, DownstreamPredictions]:
        """
        Perform a training/validation/testing step.

        Args:
        - batch: Input data batch.
        - stage (str): Stage of operation (train/val/test).

        Returns:
        - Tuple containing loss value, intensity predictions, and downstream predictions.
        """
        embeddings = self.forward(batch)
        intensity_prediction, downstream_predictions = self.joined_head(embeddings, batch)

        return (
            intensity_prediction.loss + downstream_predictions.loss
            if self.downstream_head.additive_loss_component
            else intensity_prediction.loss,
            intensity_prediction,
            downstream_predictions
        )

    def log_downstream(
            self,
            downstream_predictions: DownstreamPredictions,
            stage: str
    ) -> None:
        """
        Log downstream predictions' metrics for a given stage.

        Args:
        - downstream_predictions (DownstreamPredictions): Predictions from downstream.
        - stage (str): Stage of operation (train/val/test).
        """
        for key in downstream_predictions.metrics.keys():
            self.log(
                f"{stage}/{key}",
                downstream_predictions.metrics[key],
                on_step=True if stage == "train" else False,
                on_epoch=False if stage == "train" else True,
                prog_bar=True
            )

    def run_evaluation(self, batch: Any, stage: str) -> torch.Tensor:
        """
        Evaluation step for training, validation, or test.

        Args:
        - batch: Input data batch.
        - stage: Stage of operation (train/val/test).

        Returns:
        - torch.Tensor: Loss value obtained during evaluation.
        """
        loss, intensity_prediction, downstream_predictions = self.step(batch, stage)
        self.log(f"{stage}/log_likelihood", -intensity_prediction.loss, on_step=stage == "train",
                 on_epoch=stage != "train", prog_bar=False)
        self.log_downstream(downstream_predictions, stage)
        self.log(f"{stage}/loss", loss, on_step=stage == "train", on_epoch=stage != "train", prog_bar=False)

        return loss

    def training_step(self, batch: Any, batch_idx: int):
        """
        Training step.

        Args:
        - batch: Input data batch.
        - batch_idx: Index of the current batch.

        Returns:
        - Dictionary with the loss value.
        """
        loss = self.run_evaluation(batch, "train")
        return {"loss": loss}

    def validation_step(self, batch: Any, batch_idx: int):
        """
        Validation step.

        Args:
        - batch: Input data batch.
        - batch_idx: Index of the current batch.
        """
        self.run_evaluation(batch, "val")

    def test_step(self, batch: Any, batch_idx: int):
        """
        Test step.

        Args:
        - batch: Input data batch.
        - batch_idx: Index of the current batch.
        """
        self.run_evaluation(batch, "test")

    def configure_optimizers(self):
        """
        Configure optimizers for training.

        Returns:
        - Dictionary containing optimizer and optional learning rate scheduler.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            scheduler_config = self.hparams.scheduler_monitoring_params
            scheduler_config["scheduler"] = scheduler

            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler_config,
            }
        return {"optimizer": optimizer}
