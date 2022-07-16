from abc import ABC, abstractmethod

import torch
from pytorch_lightning import LightningModule

from typing import Tuple, Union


class MetricsCore(ABC):
    """
    Core class for metrics computation. Stores predictions and then returns all the metrics.
    """
    def __init__(
        self,
        return_time_metric,
        event_type_metric
    ):
        """
        args:
            return_time_metric - metric for return times, takes y_pred, y_true as args
            event_type_metric - metric for event types, takes y_pred, y_true as args
        """
        self.return_time_meric = return_time_metric
        self.event_type_metric = event_type_metric
    
    @property
    def return_time_target(self) -> torch.Tensor:
        """
        Return time target 1d torch Tensor
        """
        return self.__return_time_target
    
    @property
    def event_type_target(self) -> torch.Tensor:
        """
        Event type target 1d torch Tensor
        """
        return self.__event_type_target
    
    @property
    def return_time_predicted(self) -> torch.Tensor:
        """
        Return time prediction 1d torch Tensor
        """
        return self.__return_time_predicted
    
    @property
    def event_type_predicted(self) -> torch.Tensor:
        """
        Event type prediction 1d torch Tensor
        """
        return self.__event_type_predicted
    
    @property
    def step_return_time_target(self) -> torch.Tensor:
        """
        Current step return time target 1d torch Tensor
        """
        return self.__step_return_time_target
    
    @property
    def step_event_type_target(self) -> torch.Tensor:
        """
        Current step event type target 1d torch Tensor
        """
        return self.__step_event_type_target
    
    @property
    def step_return_time_predicted(self) -> torch.Tensor:
        """
        Current step return time prediction 1d torch Tensor
        """
        return self.__step_return_time_predicted
    
    @property
    def step_event_type_predicted(self) -> torch.Tensor:
        """
        Current step event type prediction 1d torch Tensor
        """
        return self.__step_event_type_predicted
    
    @staticmethod
    @abstractmethod
    def get_return_time_target(
        inputs: Union[Tuple[...], torch.Tensor]
    ) -> torch.Tensor:
        """
        Takes input batch and returns the corresponding return time targets as 1d Tensor
        
        args:
            inputs - Tuple or torch.Tensor, batch received from the dataloader
        
        return:
            return_time_target - torch.Tensor, 1d Tensor with return time targets
        """
        return
    
    @staticmethod
    @abstractmethod
    def get_event_type_target(
        inputs: Union[Tuple[...], torch.Tensor]
    ) -> torch.Tensor:
        """
        Takes input batch and returns the corresponding event type targets as 1d Tensor
        
        args:
            inputs - Tuple or torch.Tensor, batch received from the dataloader
        
        return:
            event_type_target - torch.Tensor, 1d Tensor with event type targets
        """
        return
    
    @staticmethod
    @abstractmethod
    def get_return_time_predicted(
        pl_module: LightningModule,
        inputs: Union[Tuple[...], torch.Tensor],
        outputs: Union[Tuple[...], torch.Tensor]
    ) -> torch.Tensor:
        """
        Takes input batch and returns the corresponding event type targets as 1d Tensor
        
        args:
            pl_module - LightningModule, training lightning model
            inputs - Tuple or torch.Tensor, batch received from the dataloader
            outputs - Tuple or torch.Tensor, model output
        
        return:
            return_time_predicted - torch.Tensor, 1d Tensor with return time prediction
        """
        return
    
    @staticmethod
    @abstractmethod
    def get_event_type_predicted(
        pl_module: LightningModule,
        inputs: Union[Tuple[...], torch.Tensor],
        outputs: Union[Tuple[...], torch.Tensor]
    ) -> torch.Tensor:
        """
        Takes input batch and returns the corresponding event type targets as 1d Tensor
        
        args:
            pl_module - LightningModule, training lightning model
            inputs - Tuple or torch.Tensor, batch received from the dataloader
            outputs - Tuple or torch.Tensor, model output
        
        return:
            event_type_predicted - torch.Tensor, 1d Tensor with event type prediction
        """
        return
    
    def comput_loss_and_add_values(
        self,
        pl_module: LightningModule,
        inputs: Union[Tuple[...], torch.Tensor],
        outputs: Union[Tuple[...], torch.Tensor]
    ) -> 
    
    def compute_metrics(
        self,
        pl_module: LightningModule,
        inputs: Union[Tuple[...], torch.Tensor],
        outputs: Union[Tuple[...], torch.Tensor]
    ):
        loss_value = self.compute_loss(pl_module, inputs, outputs)
        