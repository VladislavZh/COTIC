from abc import ABC, abstractmethod
import inspect

import torch
from pytorch_lightning import LightningModule

from typing import Tuple, Union, Optional, Any


class MetricsCore(ABC):
    """Core class for metrics computation. Stores predictions and then returns all the metrics."""

    def __init__(self, return_time_metric, event_type_metric) -> None:
        """
        Initialize metrics core.

        :param return_time_metric: metric for return times, takes y_pred, y_true as args
        :param event_type_metric: metric for event types, takes y_pred, y_true as args
        """
        self.__save_init_params()

        self.return_time_metric = return_time_metric
        self.event_type_metric = event_type_metric
        self.input_denorm = torch.zeros(100, 100)
        self.output_denorm = torch.zeros(100, 100)
        self.__return_time_target = torch.Tensor([])
        self.__event_type_target = torch.Tensor([])
        self.__return_time_predicted = torch.Tensor([])
        self.__event_type_predicted = torch.Tensor([])
        self.__ll_per_event = torch.Tensor([])

    def copy_empty(self):
        """Returns the object of the same type with the same initial parameters."""
        return type(self)(**self.__init_params)

    def __save_init_params(self) -> None:
        """Stores init args."""
        current_frame = inspect.currentframe()
        frame = current_frame.f_back.f_back
        _, _, _, local_vars = inspect.getargvalues(frame)
        del local_vars["self"]
        del local_vars["__class__"]
        self.__init_params = local_vars

    @property
    def return_time_target(self) -> torch.Tensor:
        """Return time target 1d torch Tensor."""
        return self.__return_time_target

    @property
    def event_type_target(self) -> torch.Tensor:
        """Event type target 1d torch Tensor."""
        return self.__event_type_target

    @property
    def return_time_predicted(self) -> torch.Tensor:
        """Return time prediction 1d torch Tensor."""
        return self.__return_time_predicted

    @property
    def event_type_predicted(self) -> torch.Tensor:
        """Event type unnormalized predictions 2d torch Tensor."""
        return self.__event_type_predicted

    @property
    def ll_per_event(self) -> torch.Tensor:
        """Log likelihood per event for each sequence 1d torch Tensor."""
        return self.__ll_per_event

    @property
    def step_return_time_target(self) -> torch.Tensor:
        """Current step return time target 1d torch Tensor."""
        return self.__step_return_time_target

    @property
    def step_event_type_target(self) -> torch.Tensor:
        """Current step event type target 1d torch Tensor."""
        return self.__step_event_type_target

    @property
    def step_return_time_predicted(self) -> torch.Tensor:
        """Current step return time prediction 1d torch Tensor."""
        return self.__step_return_time_predicted

    @property
    def step_event_type_predicted(self) -> torch.Tensor:
        """Current step event type unnormalized predictions 2d torch Tensor."""
        return self.__step_event_type_predicted

    @property
    def step_ll_per_event(self) -> torch.Tensor:
        """Current step log likelihood per event for each sequence 1d torch Tensor."""
        return self.__step_ll_per_event

    @staticmethod
    @abstractmethod
    def get_return_time_target(inputs: Union[Tuple, torch.Tensor]) -> torch.Tensor:
        """Takes input batch and returns the corresponding return time targets as 1d Tensor.

        :param inputs: batch received from the dataloader (Tuple or torch.Tensor)

        :return: return_time_target - 1d Tensor with return time targets
        """
        return

    @staticmethod
    @abstractmethod
    def get_event_type_target(inputs: Union[Tuple, torch.Tensor]) -> torch.Tensor:
        """Take input batch and returns the corresponding event type targets as 1d Tensor.

        :param inputs: batch received from the dataloader (Tuple or torch.Tensor)

        :return: event_type_target - 1d Tensor with event type target
        """
        return

    @staticmethod
    @abstractmethod
    def get_return_time_predicted(
        pl_module: LightningModule,
        inputs: Union[Tuple, torch.Tensor],
        outputs: Union[Tuple, torch.Tensor],
    ) -> torch.Tensor:
        """Take lighning model, input batch and model outputs, returns the corresponding predicted return times as 1d Tensor

        :param pl_module: LightningModule, training lightning model
        :param inputs: Tuple or torch.Tensor, batch received from the dataloader
        :param outputs: Tuple or torch.Tensor, model output

        :return: return_time_predicted - 1d Tensor with return time prediction
        """
        return

    @staticmethod
    @abstractmethod
    def get_event_type_predicted(
        pl_module: LightningModule,
        inputs: Union[Tuple, torch.Tensor],
        outputs: Union[Tuple, torch.Tensor],
    ) -> torch.Tensor:
        """Take lighning model, input batch and model outputs, returns the corresponding predicted event types as 1d Tensor

        :param pl_module: LightningModule, training lightning model
        :param inputs: Tuple or torch.Tensor, batch received from the dataloader
        :param outputs: Tuple or torch.Tensor, model output

        :return: event_type_predicted - 2d Tensor with event type unnormalized predictions
        """
        return

    @abstractmethod
    def compute_log_likelihood_per_event(
        self,
        pl_module: LightningModule,
        inputs: Union[Tuple, torch.Tensor],
        outputs: Union[Tuple, torch.Tensor],
    ) -> torch.Tensor:
        """Compute log likelihood per event for each sequence in the batch as 1d Tensor of shape (bs, ).

        :param pl_module: LightningModule, training lightning model
        :param inputs: Tuple or torch.Tensor, batch received from the dataloader
        :param outputs: Tuple or torch.Tensor, model output

        :return: log_likelihood_per_seq - 1d Tensor with log likelihood per event prediction, shape = (bs,)
        """
        return

    @abstractmethod
    def compute_loss(
        self,
        pl_module: LightningModule,
        inputs: Union[Tuple, torch.Tensor],
        outputs: Union[Tuple, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute loss for backpropagation.

        :param pl_module: LightningModule, training lightning model
        :param inputs: Tuple or torch.Tensor, batch received from the dataloader
        :param outputs: Tuple or torch.Tensor, model output

        :return: loss - torch.Tensor, loss for backpropagation
        """
        return

    def __check_shapes(self) -> None:
        """Check shapes of all the predictions and targets."""
        if len(self.__step_return_time_target.shape) != 1:
            raise ValueError(
                f"Wrong return time target shape. Expected 1, got {len(self.__step_return_time_target.shape)}"
            )
        if len(self.__step_event_type_target.shape) != 1:
            raise ValueError(
                f"Wrong event type target shape. Expected 1, got {len(self.__step_event_type_target.shape)}"
            )
        if len(self.__step_return_time_predicted.shape) != 1:
            raise ValueError(
                f"Wrong predicted return time shape. Expected 1, got {len(self.__step_return_time_predicted.shape)}"
            )
        if len(self.__step_event_type_predicted.shape) != 2:
            raise ValueError(
                f"Wrong predicted event type shape. Expected 2, got {len(self.__step_event_type_predicted.shape)}"
            )
        if len(self.__step_ll_per_event.shape) != 1:
            raise ValueError(
                f"Wrong log likelihood shape. Expected 1, got {len(self.__step_ll_per_event.shape)}"
            )

    def __append_step_values(self):
        """Append current step predictions and targets to the corresponding tensors."""
        self.__return_time_target = torch.concat(
            [
                self.__return_time_target,
                self.__step_return_time_target.detach().clone().cpu(),
            ]
        )
        self.__event_type_target = torch.concat(
            [
                self.__event_type_target,
                self.__step_event_type_target.detach().clone().cpu(),
            ]
        )
        self.__return_time_predicted = torch.concat(
            [
                self.__return_time_predicted,
                self.__step_return_time_predicted.detach().clone().cpu(),
            ]
        )
        self.__event_type_predicted = torch.concat(
            [
                self.__event_type_predicted,
                self.__step_event_type_predicted.detach().clone().cpu(),
            ]
        )
        self.__ll_per_event = torch.concat(
            [self.__ll_per_event, self.__step_ll_per_event.detach().clone().cpu()]
        )

    def compute_loss_and_add_values(
        self,
        pl_module: LightningModule,
        inputs: Union[Tuple, torch.Tensor],
        outputs: Union[Tuple, torch.Tensor],
        scaler: Optional[Any] = None,
    ) -> torch.Tensor:
        """Add step targets and predictions, computes loss

        :param pl_module: LightningModule, training lightning model
        :param inputs: Tuple or torch.Tensor, batch received from the dataloader
        :param outputs: Tuple or torch.Tensor, model output

        :return: loss - loss for backpropagation
        """
        # getting first because of tensor usability
        loss = self.compute_loss(pl_module, inputs, outputs)
        if scaler is not None:
            self.input_denorm = inputs
            self.output_denorm = outputs
            self.input_denorm[0][:] = scaler.denormalization(inputs[0])
            self.output_denorm[1][1][:] = scaler.denormalization(outputs[1][1])
        else:
            self.input_denorm = inputs
            self.output_denorm = outputs

        self.__step_return_time_target = self.get_return_time_target(self.input_denorm)
        self.__step_event_type_target = self.get_event_type_target(self.input_denorm)
        self.__step_return_time_predicted = self.get_return_time_predicted(
            pl_module, self.input_denorm, self.output_denorm
        )
        self.__step_event_type_predicted = self.get_event_type_predicted(
            pl_module, self.input_denorm, self.output_denorm
        )
        self.__step_ll_per_event = self.compute_log_likelihood_per_event(
            pl_module, self.input_denorm, self.output_denorm
        )

        self.__check_shapes()

        self.__append_step_values()

        loss = self.compute_loss(pl_module, inputs, outputs)

        return loss

    def compute_metrics(self) -> Tuple[float, float, float]:
        """Returns mean log likelihood per event, return time metric value and event type metric value."""
        ll = torch.mean(self.ll_per_event)
        return_time_metric = self.return_time_metric(
            self.return_time_predicted, self.return_time_target
        )
        event_type_metric = self.event_type_metric(
            torch.nn.functional.softmax(self.event_type_predicted, dim=1),
            self.event_type_target,
        )
        return ll, return_time_metric, event_type_metric

    def clear_values(self):
        """Clears stored values."""
        self.__return_time_target = torch.Tensor([])
        self.__event_type_target = torch.Tensor([])
        self.__return_time_predicted = torch.Tensor([])
        self.__event_type_predicted = torch.Tensor([])
        self.__ll_per_event = torch.Tensor([])
