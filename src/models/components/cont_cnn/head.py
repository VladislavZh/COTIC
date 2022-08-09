import torch
import torch.nn as nn

from typing import Tuple

class PredictionHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_types: int
    ) -> None:
        super().__init__()
        self.return_time_prediction = nn.Sequential(nn.Linear(in_channels, 128),nn.ReLU(),nn.Linear(128,1))
        self.event_type_prediction = nn.Sequential(nn.Linear(in_channels, 128),nn.ReLU(),nn.Linear(128,num_types))
        
    def forward(
        self,
        enc_output: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.return_time_prediction(enc_output), self.event_type_prediction(enc_output)


class IntensityBasedHead(nn.Module):
    def __init__(
        self,
        max_val: float,
        sim_size: int,
    ) -> None:
        super().__init__()
        self.max_val = max_val
        self.sim_size = sim_size
        
    @staticmethod
    def __add_sim_times(
        times: torch.Tensor,
        max_val: float,
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
        sim_delta_times = (torch.rand(list(delta_times.shape)+[sim_size]).to(times.device) * max_val).sort(dim=2).values
        full_times = torch.concat([sim_delta_times.to(times.device),delta_times.unsqueeze(2)], dim = 2)
        full_times = full_times + times[:,:-1].unsqueeze(2)
        full_times[delta_times<0,:] = 0
        full_times = full_times.flatten(1)
        bos_full_times = torch.concat([torch.zeros(times.shape[0],1).to(times.device), full_times], dim = 1)
        
        return bos_full_times
    
    def __simulate_and_compute_times_and_events(
        self,
        times,
        events,
        enc_output,
        pl_module
    ):
        """ Log-likelihood of non-events, using Monte Carlo integration. """
        
        bos_full_times = self.__add_sim_times(times, self.max_val, self.sim_size)
        all_lambda = pl_module.net.final(bos_full_times, times, enc_output, events.ne(0), self.sim_size) # shape = (bs, (num_samples + 1) * L + 1, num_types)
        
        bs, _, num_types = all_lambda.shape
        
        between_lambda = all_lambda.transpose(1,2)[:,:,1:].reshape(bs, num_types, times.shape[1]-1, self.sim_size + 1)[...,:-1]

        diff_time = bos_full_times[:, :-1].reshape(bs, -1, self.sim_size + 1)
        diff_time = diff_time - diff_time[...,0].clone().unsqueeze(-1) # shape = (bs, L, sim_size + 1)
        diff_time = diff_time[...,1:] # shape = (bs, L, sim_size)
        
        between_lambda_all = torch.sum(between_lambda, dim = 1)
        
        lambda_int = torch.cumsum(between_lambda_all, dim = -1)/torch.arange(1, self.sim_size + 1)[None,None,:].to(times.device)*diff_time
        predicted_time = torch.sum(torch.exp(-lambda_int),dim=-1)*diff_time[...,-1]/self.sim_size
        
        bos_full_times = torch.concat([(times[:,:-1] + predicted_time).unsqueeze(-1), times[:,1:].unsqueeze(-1)], dim = -1).flatten(1)
        bos_full_times = torch.concat([torch.zeros(bs, 1).to(times.device), bos_full_times], dim=-1)
        all_lambda = pl_module.net.final(bos_full_times, times, enc_output, events.ne(0), 1) # shape = (bs, (num_samples + 1) * L + 1, num_types)
        
        predicted_event = torch.log(all_lambda.transpose(1,2)[:,:,1:].reshape(bs, num_types, times.shape[1]-1, 2)[...,0].transpose(1,2) + 1e-9)
        
        predicted_time = torch.concat([predicted_time, torch.zeros(bs,1).to(times.device)], dim=-1)
        predicted_event = torch.concat([predicted_event, torch.zeros(bs,1,num_types).to(times.device)], dim=1)
        
        return predicted_time, predicted_event
        
    def __add_bos(self, event_times, event_types, lengths):
        bs, L = event_times.shape
        event_times = torch.concat([torch.zeros(bs, 1).to(event_times.device), event_times], dim = 1)
        max_event_type = torch.max(event_types) + 1
        tmp = (torch.ones(bs,1).to(event_types.device) * max_event_type).long()
        event_types = torch.concat([tmp, event_types], dim = 1)
        lengths += 1
        return event_times, event_types, lengths
    
    def forward(
        self,
        event_times,
        event_types,
        enc_output,
        pl_module
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        lengths = torch.sum(event_types.ne(0).type(torch.float), dim = 1).long()
        event_times, event_types, lengths = self.__add_bos(event_times, event_types, lengths)
        return self.__simulate_and_compute_times_and_events(event_times, event_types, enc_output, pl_module)
