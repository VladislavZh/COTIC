from typing import Any
import torch
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric

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


        self.log_likelihood_metric = {
            'train': MeanMetric(),
            'val': MeanMetric(),
            'test': MeanMetric()
        }
        self.return_time_mae = {
            'train': MeanMetric(),
            'val': MeanMetric(),
            'test': MeanMetric()
        }
        self.event_type_accuracy = {
            'train': MeanMetric(),
            'val': MeanMetric(),
            'test': MeanMetric()
        }
        self.best_log_likelihood = MaxMetric()

    def reset_all_metrics(self, stage: str):
        self.log_likelihood_metric[stage].reset()
        self.return_time_mae[stage].reset()
        self.event_type_accuracy[stage].reset()

    def on_train_start(self) -> None:
        self.reset_all_metrics('val')
        self.best_log_likelihood.reset()

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

    def log_metrics(
            self,
            log_likelihood: float,
            return_time_mae: float,
            event_type_accuracy: float,
            stage: str
    ) -> None:
        """
        Log downstream predictions' metrics for a given stage.
        """
        self.log(f"{stage}/log_likelihood", log_likelihood, on_step=False, on_epoch=True, prog_bar=True)
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

        self.log_likelihood_metric[stage].update(-negative_log_likelihood.cpu())
        self.return_time_mae[stage].update(return_time_mae)
        self.event_type_accuracy[stage].update(event_type_accuracy)

        self.log(f"{stage}/loss", loss, on_step=stage == "train", on_epoch=stage != "train", prog_bar=True)

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

    def on_train_epoch_end(self) -> None:
        self.log_metrics(
            self.log_likelihood_metric['train'].compute().item(),
            self.return_time_mae['train'].compute().item(),
            self.event_type_accuracy['train'].compute().item(),
            'train'
        )
        self.reset_all_metrics('train')

    def validation_step(self, batch: Any, batch_idx: int):
        """
        Validation step.

        Args:
        - batch: Input data_utils batch.
        - batch_idx: Index of the current batch.
        """
        self.run_evaluation(batch, "val")

    def on_validation_epoch_end(self) -> None:
        self.log_metrics(
            self.log_likelihood_metric['val'].compute().item(),
            self.return_time_mae['val'].compute().item(),
            self.event_type_accuracy['val'].compute().item(),
            'val'
        )
        self.best_log_likelihood.update(self.log_likelihood_metric['val'].compute())
        self.log(f"val/best_log_likelihood", self.best_log_likelihood.compute(), on_step=False, on_epoch=True, prog_bar=True)
        self.reset_all_metrics('val')

    def test_step(self, batch: Any, batch_idx: int):
        """
        Test step.

        Args:
        - batch: Input data_utils batch.
        - batch_idx: Index of the current batch.
        """
        self.run_evaluation(batch, "test")

    def on_test_epoch_end(self) -> None:
        self.log_metrics(
            self.log_likelihood_metric['test'].compute().item(),
            self.return_time_mae['test'].compute().item(),
            self.event_type_accuracy['test'].compute().item(),
            'test'
        )
        self.reset_all_metrics('test')

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
