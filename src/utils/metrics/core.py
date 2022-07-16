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
        
        self.__return_time_target    = torch.Tensor([])
        self.__event_type_target     = torch.Tensor([])
        self.__return_time_predicted = torch.Tensor([])
        self.__event_type_predicted  = torch.Tensor([])
        self.__ll_per_event          = torch.Tensor([])
        
    
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
    def ll_per_event(self) -> torch.Tensor:
        """
        Log likelihood per event for each sequence 1d torch Tensor
        """
        return self.__ll_per_event
    
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
    
    @property
    def step_ll_per_event(self) -> torch.Tensor:
        """
        Current step log likelihood per event for each sequence 1d torch Tensor
        """
        return self.__step_ll_per_event
    
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
        Takes lighning model, input batch and model outputs, returns the corresponding predicted return times as 1d Tensor
        
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
        Takes lighning model, input batch and model outputs, returns the corresponding predicted event types as 1d Tensor
        
        args:
            pl_module - LightningModule, training lightning model
            inputs - Tuple or torch.Tensor, batch received from the dataloader
            outputs - Tuple or torch.Tensor, model output
        
        return:
            event_type_predicted - torch.Tensor, 1d Tensor with event type prediction
        """
        return
    
    @abstractmethod
    def compute_log_likelihood_per_event(
        self,
        pl_module: LightningModule,
        inputs: Union[Tuple[...], torch.Tensor],
        outputs: Union[Tuple[...], torch.Tensor]
    ) -> torch.Tensor:
        """
        Takes lighning model, input batch and model outputs, returns the corresponding log likelihood per event for each sequence in the batch as 1d Tensor of shape (bs,), 
        one can use self.step_[return_time_target/event_type_target/return_time_predicted/event_type_predicted] if needed
        
        args:
            pl_module - LightningModule, training lightning model
            inputs - Tuple or torch.Tensor, batch received from the dataloader
            outputs - Tuple or torch.Tensor, model output
        
        return:
            log_likelihood_per_seq - torch.Tensor, 1d Tensor with log likelihood per event prediction, shape = (bs,)
        """
        return
    
    @abstractmethod
    def compute_loss(
        self,
        pl_module: LightningModule,
        inputs: Union[Tuple[...], torch.Tensor],
        outputs: Union[Tuple[...], torch.Tensor]
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
        return
        
    def __check_shapes(self):
        if len(self.__step_return_time_target.shape) != 1:
            raise ValueError(f'Wrong return time target shape. Expected 1, got {len(self.__step_return_time_target.shape)}')
        if len(self.__step_event_type_target.shape) != 1:
            raise ValueError(f'Wrong event type target shape. Expected 1, got {len(self.__step_event_type_target.shape)}')
        if len(self.__step_return_time_predicted.shape) != 1:
            raise ValueError(f'Wrong predicted return time shape. Expected 1, got {len(self.__step_return_time_predicted.shape)}')
        if len(self.__step_event_type_predicted.shape) != 1:
            raise ValueError(f'Wrong predicted event type shape. Expected 1, got {len(self.__step_event_type_predicted.shape)}')
        if len(self.__step_ll_per_event.shape) != 1:
            raise ValueError(f'Wrong log likelihood shape. Expected 1, got {len(self.__step_ll_per_event.shape)}')
            
    def __append_step_values(self):
        self.__return_time_target = torch.concat([
            self.__return_time_target,
            self.__step_return_time_target.detach().clone()
        ])
        self.__event_type_target = torch.concat([
            self.__event_type_target,
            self.__step_event_type_target.detach().clone()
        ])
        self.__return_time_predicted = torch.concat([
            self.__return_time_predicted,
            self.__step_return_time_predicted.detach().clone()
        ])
        self.__event_type_predicted = torch.concat([
            self.__event_type_predicted,
            self.__step_event_type_predicted.detach().clone()
        ])
        self.__ll_per_event = torch.concat([
            self.__ll_per_event,
            self.__step_ll_per_event.detach().clone()
        ])
    
    def compute_loss_and_add_values(
        self,
        pl_module: LightningModule,
        inputs: Union[Tuple[...], torch.Tensor],
        outputs: Union[Tuple[...], torch.Tensor]
    ) -> torch.Tensor:
        """
        Takes model, inputs and outputs, adds step targets and predictions and computes loss
        
        args:
            pl_module - LightningModule, training lightning model
            inputs - Tuple or torch.Tensor, batch received from the dataloader
            outputs - Tuple or torch.Tensor, model output
        
        return:
            loss - torch.Tensor, loss for backpropagation
        """
        self.__step_return_time_target    = self.get_return_time_target(inputs)
        self.__step_event_type_target     = self.get_event_type_target(inputs)
        self.__step_return_time_predicted = self.get_return_time_predicted(pl_module, inputs, outputs)
        self.__step_event_type_predicted  = self.get_event_type_predicted(pl_module, inputs, outputs)
        self.__step_ll_per_event          = self.compute_log_likelihood_per_event(pl_module, inputs, outputs)
        
        self.__check_shapes()
        
        self.__append_step_values()
        
        loss = self.compute_loss(pl_module, inputs, outputs)
        
        return loss

    def compute_metrics(
        self
    ) -> Tuple[float, float, float]:
        """
        Returns mean log likelihood per event, return time metric value and event type metric value
        """
        ll = torch.mean(self.ll_per_event)
        return_time_metric = self.return_time_meric(self.return_time_predicted, self.return_time_target)
        event_type_metric  = self.event_type_metric(self.event_type_predicted, self.event_type_target)
        return ll, return_time_metric, event_type_metric
        
    def clear_values(self):
        """
        Clears stored values
        """
        self.__return_time_target    = torch.Tensor([])
        self.__event_type_target     = torch.Tensor([])
        self.__return_time_predicted = torch.Tensor([])
        self.__event_type_predicted  = torch.Tensor([])
        self.__ll_per_event          = torch.Tensor([])
        