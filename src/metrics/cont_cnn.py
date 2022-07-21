import torch
from pytorch_lightning import LightningModule
import math

from src.utils.metrics import MetricsCore
from typing import Union, Tuple


class CCNNMetrics(MetricsCore):
    def __init__(
        self,
        return_time_metric,
        event_type_metric,
        sim_size = 100
    ):
        super().__init__(return_time_metric, event_type_metric)
        self.type_loss_func = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
        self.return_time_loss_func = torch.nn.MSELoss()
        self.sim_size = sim_size
        
    @staticmethod
    def get_return_time_target(
        inputs: Union[Tuple, torch.Tensor]
    ) -> torch.Tensor:
        """
        Takes input batch and returns the corresponding return time targets as 1d Tensor
        
        args:
            inputs - Tuple or torch.Tensor, batch received from the dataloader
        
        return:
            return_time_target - torch.Tensor, 1d Tensor with return time targets
        """
        event_time = inputs[0]
        return_time = event_time[:,1:] - event_time[:,:-1]
        mask = inputs[1].ne(0)[:,1:]
        return return_time[mask]
    
    @staticmethod
    def get_event_type_target(
        inputs: Union[Tuple, torch.Tensor]
    ) -> torch.Tensor:
        """
        Takes input batch and returns the corresponding event type targets as 1d Tensor
        
        args:
            inputs - Tuple or torch.Tensor, batch received from the dataloader
        
        return:
            event_type_target - torch.Tensor, 1d Tensor with event type targets
        """
        event_type = inputs[1][:,1:]
        mask = inputs[1].ne(0)[:,1:]
        return event_type[mask]
    
    @staticmethod
    def get_return_time_predicted(
        pl_module: LightningModule,
        inputs: Union[Tuple, torch.Tensor],
        outputs: Union[Tuple, torch.Tensor]
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
        return_time_prediction = outputs[1][0].squeeze_(-1)[:,:-1]
        mask = inputs[1].ne(0)[:,1:]
        return return_time_prediction[mask]
    
    @staticmethod
    def get_event_type_predicted(
        pl_module: LightningModule,
        inputs: Union[Tuple, torch.Tensor],
        outputs: Union[Tuple, torch.Tensor]
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
        event_type_prediction = outputs[1][1][:,:-1,:]
        mask = inputs[1].ne(0)[:,1:]
        return event_type_prediction[mask,:]
    
    @staticmethod
    def compute_event(type_lambda, non_pad_mask):
        """ Log-likelihood of events. """

        # add 1e-9 in case some events have 0 likelihood
        type_lambda += math.pow(10, -9)
        type_lambda.masked_fill_(~non_pad_mask.bool(), 1.0)

        result = torch.log(type_lambda)
        return result
    
    @staticmethod
    def __add_sim_times(
        times: torch.Tensor,
        features: torch.Tensor,
        sim_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Takes batch of times and events and adds sim_size auxiliar times
        
        args:
            times  - torch.Tensor of shape (bs, max_len), that represents event arrival time since start
            featuers - torch.Tensor of shape (bs, max_len, d), that represents event features
            sim_size - int, number of points to simulate
            
        returns:
            bos_full_times  - torch.Tensor of shape(bs, (sim_size + 1) * (max_len - 1) + 1) that consists of times and sim_size auxiliar times between events
            bos_full_features - torch.Tensor of shape(bs, (sim_size + 1) * (max_len - 1) + 1, d) that consists of event features and sim_size zeros between events
        """
        delta_times = times[:,1:] - times[:,:-1]
        sim_delta_times = (torch.rand(list(delta_times.shape)+[sim_size]) * delta_times.unsqueeze(2)).sort(dim=2).values
        full_times = torch.concat([sim_delta_times,delta_times.unsqueeze(2)], dim = 2)
        full_times = full_times + times[:,:-1].unsqueeze(2)
        full_times[delta_times<0,:] = 0
        full_times = full_times.flatten(1)
        bos_full_times = torch.concat([torch.zeros(times.shape[0],1), full_times], dim = 1)
        
        full_events = torch.concat([torch.zeros(list(delta_times.shape)+[features.shape[2],sim_size]), features[:,1:].unsqueeze(3)], dim=3)
        bs, L, d, f_sim = full_events.shape
        full_events = full_events.transpose(2,3)
        full_events = full_events.reshape(bs, L*f_sim, d)
        bos_full_events = torch.concat([features[:,0,:].unsqueeze(1), full_events], dim = 1)
        return bos_full_times, bos_full_events
    
    @classmethod
    def compute_integral_unbiased(cls, model, enc_output, event_time, lengths, non_pad_mask, type_mask, num_samples):
        """ Log-likelihood of non-events, using Monte Carlo integration. """
        
        bos_full_times, bos_full_enc_output = cls.__add_sim_times(event_time, enc_output, num_samples)
        true_ids_template = torch.Tensor([(num_samples + 1) * i for i in range(event_time.shape[1])]).long()
        true_ids_mask = torch.zeros_like(bos_full_times)
        for i in range(bos_full_times.shape[0]):
             true_ids_mask[i,true_ids_template[:lengths[i]]] = 1
            true_ids_mask = true_ids_mask.bool()

        diff_time = (event_time[:, 1:] - event_time[:, :-1]) * non_pad_mask[:, 1:]
        temp_time = diff_time.unsqueeze(2) * \
                    torch.rand([*diff_time.size(), num_samples], device=enc_output.device)

        all_lambda = model.get_lambdas(event_time[:,:-1], enc_output[:,:-1,:], temp_time, non_pad_mask[:,1:])
        all_lambda = torch.sum(all_lambda, dim=(2,3)) / num_samples

        unbiased_integral = all_lambda * diff_time
        return unbiased_integral
    
    def event_and_non_event_log_likelihood(
        self,
        pl_module: LightningModule,
        enc_output: torch.Tensor,
        event_time: torch.Tensor,
        event_type: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes log of the intensity and the integral
        """

        non_pad_mask = event_type.ne(0).type(torch.float)

        type_mask = torch.zeros([*event_type.size(), pl_module.net.num_types], device=enc_output.device)
        for i in range(pl_module.net.num_types):
            type_mask[:, :, i] = (event_type == i + 1).bool().to(enc_output.device)
        
        event_time = torch.concat([torch.zeros(event_time.shape[0],1).to(event_time.device), event_time], dim = 1)
        lengths = (torch.sum(inputs[1].ne(0).type(torch.float), dim = 1) + 1).long()
        true_ids = torch.arange(event_time.shape[1])[None,:].repeat(event_time.shape[0], 1)
        true_ids = (true_ids < lengths[:, None])
        all_lambda = pl_module.net.final(event_time, enc_output, lengths.to(enc_output.device), true_ids, 0)

        type_lambda = torch.sum(all_lambda[:,1:,:] * type_mask, dim=2) #shape = (bs, L)

        # event log-likelihood
        event_ll = self.compute_event(type_lambda, non_pad_mask)
        event_ll = torch.sum(event_ll, dim=-1)

        # non-event log-likelihood, MC integration
        non_event_ll = self.compute_integral_unbiased(pl_module.net, enc_output, event_time, lengths, non_pad_mask, type_mask, self.sim_size)
        non_event_ll = torch.sum(non_event_ll, dim=-1)

        return event_ll, non_event_ll
    
    def compute_log_likelihood_per_event(
        self,
        pl_module: LightningModule,
        inputs: Union[Tuple, torch.Tensor],
        outputs: Union[Tuple, torch.Tensor]
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
        event_ll, non_event_ll = self.event_and_non_event_log_likelihood(
            pl_module,
            outputs[0],
            inputs[0],
            inputs[1]
        )
        lengths = torch.sum(inputs[1].ne(0).type(torch.float), dim = 1)
        results = (event_ll - non_event_ll)/lengths
        return results
    
    def type_loss(self, prediction, types):
        """ Event prediction loss, cross entropy. """

        # convert [1,2,3] based types to [0,1,2]; also convert padding events to -1
        truth = types[:, 1:] - 1
        prediction = prediction[:, :-1, :]

        loss = self.type_loss_func(prediction.transpose(1, 2), truth)

        loss = torch.sum(loss)
        return loss
    
    def time_loss(self, prediction, event_time, event_type):
        """ Time prediction loss. """

        prediction.squeeze_(-1)
        
        mask = event_type.ne(0)[:,1:]

        true = event_time[:, 1:] - event_time[:, :-1]
        prediction = prediction[:, :-1]

        return self.return_time_loss_func(true[mask], prediction[mask])
    
    def compute_loss(
        self,
        pl_module: LightningModule,
        inputs: Union[Tuple, torch.Tensor],
        outputs: Union[Tuple, torch.Tensor]
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
        event_ll, non_event_ll = self.event_and_non_event_log_likelihood(
            pl_module,
            outputs[0],
            inputs[0],
            inputs[1]
        )
        ll_loss = -torch.sum(event_ll - non_event_ll)
        type_loss = self.type_loss(outputs[1][1], inputs[1])
        time_loss = self.time_loss(outputs[1][0], inputs[0], inputs[1])
        
        return ll_loss, type_loss + time_loss
