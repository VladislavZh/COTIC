import torch
from pytorch_lightning import LightningModule
import math

from src.utils.metrics import MetricsCore
from typing import Union, Tuple


class WNMetrics(MetricsCore):
    def __init__(self, return_time_metric, event_type_metric):
        super().__init__(return_time_metric, event_type_metric)
        self.type_loss_func = torch.nn.CrossEntropyLoss(
            ignore_index=-1, reduction="none"
        )
        self.return_time_loss_func = torch.nn.MSELoss()

    @staticmethod
    def get_return_time_target(inputs: Union[Tuple, torch.Tensor]) -> torch.Tensor:
        """
        Takes input batch and returns the corresponding return time targets as 1d Tensor

        args:
            inputs - Tuple or torch.Tensor, batch received from the dataloader

        return:
            return_time_target - torch.Tensor, 1d Tensor with return time targets
        """
        event_time = inputs[0]
        return_time = event_time[:, 1:] - event_time[:, :-1]
        mask = inputs[1].ne(0)[:, 1:]
        return return_time[mask]

    @staticmethod
    def get_event_type_target(inputs: Union[Tuple, torch.Tensor]) -> torch.Tensor:
        """
        Takes input batch and returns the corresponding event type targets as 1d Tensor

        args:
            inputs - Tuple or torch.Tensor, batch received from the dataloader

        return:
            event_type_target - torch.Tensor,  1d Tensor with event type target
        """
        event_type = inputs[1][:, 1:]
        mask = inputs[1].ne(0)[:, 1:]
        return event_type[mask]

    @staticmethod
    def get_return_time_predicted(
        pl_module: LightningModule,
        inputs: Union[Tuple, torch.Tensor],
        outputs: Union[Tuple, torch.Tensor],
    ) -> torch.Tensor:
        """
        Takes lighning model, input batch and model outputs, returns the corresponding predicted return times as 1d Tensor

        args:
            pl_module - LightningModule, training lightning model
            inputs - Tuple or torch.Tensor, batch received from the dataloader
            outputs - Tuple or torch.Tensor, model output

        return:
            return_time_predicted - torch.Tensor, 1d Tensor with return time prediction
        """
        return_time_prediction = outputs[0].squeeze_(-1)[:, :-1]
        mask = inputs[1].ne(0)[:, 1:]
        return return_time_prediction[mask]

    @staticmethod
    def get_event_type_predicted(
        pl_module: LightningModule,
        inputs: Union[Tuple, torch.Tensor],
        outputs: Union[Tuple, torch.Tensor],
    ) -> torch.Tensor:
        """
        Takes lighning model, input batch and model outputs, returns the corresponding predicted event types as 1d Tensor

        args:
            pl_module - LightningModule, training lightning model
            inputs - Tuple or torch.Tensor, batch received from the dataloader
            outputs - Tuple or torch.Tensor, model output

        return:
            event_type_predicted - torch.Tensor, 2d Tensor with event type unnormalized predictions
        """
        event_type_prediction = outputs[1][:, :-1, :]
        mask = inputs[1].ne(0)[:, 1:]

        return event_type_prediction[mask, :]

    def compute_log_likelihood_per_event(
        self,
        pl_module: LightningModule,
        inputs: Union[Tuple, torch.Tensor],
        outputs: Union[Tuple, torch.Tensor],
    ) -> torch.Tensor:
        """
        Nan placeholder with bs shape
        """
        bs = inputs[0].shape[0]
        return torch.ones(bs) * torch.nan

    def type_loss(self, prediction, types):
        """Event prediction loss, cross entropy."""

        # convert [1,2,3] based types to [0,1,2]; also convert padding events to -1
        truth = types[:, 1:] - 1
        prediction = prediction[:, :-1, :]

        loss = self.type_loss_func(prediction.transpose(1, 2), truth)

        loss = torch.sum(loss)
        return loss

    def time_loss(self, prediction, event_time, event_type):
        """Time prediction loss."""

        prediction.squeeze_(-1)

        mask = event_type.ne(0)[:, 1:]

        true = event_time[:, 1:] - event_time[:, :-1]
        prediction = prediction[:, :-1]

        return self.return_time_loss_func(true[mask], prediction[mask])

    def compute_loss(
        self,
        pl_module: LightningModule,
        inputs: Union[Tuple, torch.Tensor],
        outputs: Union[Tuple, torch.Tensor],
    ) -> torch.Tensor:
        """
        Takes lighning model, input batch and model outputs, returns the corresponding loss for backpropagation,
        one can use self.step_[return_time_target/event_type_target/return_time_predicted/event_type_predicted/ll_per_event] if needed

        args:
            pl_module - LightningModule, training lightning model
            inputs - Tuple or torch.Tensor, batch received from the dataloader
            outputs - Tuple or torch.Tensor, model output

        return:
            loss - torch.Tensor, loss for backpropagation
        """
        type_loss = self.type_loss(outputs[1], inputs[1])
        time_loss = self.time_loss(outputs[0], inputs[0], inputs[1])

        return type_loss + time_loss
