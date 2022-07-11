import torch
from torch.utils.data import Dataset

from typing import List, Tuple
from collections.abc import Iterable

class DenseCCNNDset(Dataset):
    """
    Dense Dataset for Continuous CNN, takes times, events and sim_size as arguments. 
    Adds the beginning of stream event, adds additional simulated times between events for MC integral computation.
    """
    def __init__(
        self,
        times: List[torch.Tensor],
        events: List[torch.Tensor],
        sim_size: int = 5
    ):
        self.__times, self.__events, self.__lengths = self.__bos_pad(times, events)
        self.__sim_size = sim_size
        self.__true_ids = torch.Tensor([(self.__sim_size + 1) * i for i in range(self.__times.shape[1])]).long()
        
    @staticmethod
    def __bos_pad(
        times: List[torch.Tensor],
        events: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Takes list of times and events sequences, pads them and add beginning of stream event
        
        args:
            times   - list of torch.Tensor of shape = (length,), that represents event arrival time since start
            events  - list of torch.Tensor of shape = (length,), that represents event types in {0,1,...,C-1}
        
        returns:
            times   - torch.Tensor of shape (dset_size, max_len), where max_len = max({length}) + 1, padded times with additional zero in the beginning
            events  - torch.Tensor of shape (dset_size, max_len), where max_len = max({length}) + 1, padded events with shifted events
                                                                 {0,...,C-1} -> {1,...,C}, 0 as pad value and C+1 event in the beginning
            lengths - torch.Tensor of shape (dset_size,) with length+1 values
        """
        
        lengths = torch.Tensor([len(time_seq) for time_seq in times]).long()
        max_len = torch.max(lengths)
        
        tensor_times, tensor_events = torch.zeros(len(times), max_len+1), torch.zeros(len(times), max_len+1)
        
        for i, l in enumerate(lengths):
            tensor_times[i,1:l+1] = times[i]
            tensor_events[i,1:l+1] = events[i] + 1
        
        tensor_events[:,0] = torch.max(tensor_events) + 1
        lengths += 1
        
        return tensor_times, tensor_events.long(), lengths
    
    def __add_sim_times(
        self,
        times: torch.Tensor,
        events: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Takes batch of times and events and adds sim_size auxiliar times
        
        args:
            times  - torch.Tensor of shape (bs, max_len), that represents event arrival time since start
            events - torch.Tensor of shape (bs, max_len), that represents event types
            
        returns:
            bos_full_times  - torch.Tensor of shape(bs, (sim_size + 1) * (max_len - 1) + 1) that consists of times and sim_size auxiliar times between events
            bos_full_events - torch.Tensor of shape(bs, (sim_size + 1) * (max_len - 1) + 1) that consists of events and sim_size zeros between events
        """
        delta_times = times[:,1:] - times[:,:-1]
        sim_delta_times = (torch.rand(list(delta_times.shape)+[self.__sim_size]) * delta_times.unsqueeze(2)).sort(dim=2).values
        full_times = torch.concat([sim_delta_times,delta_times.unsqueeze(2)], dim = 2)
        full_times = full_times + times[:,:-1].unsqueeze(2)
        full_times[delta_times<0,:] = 0
        full_times = full_times.flatten(1)
        bos_full_times = torch.concat([torch.zeros(times.shape[0],1), full_times], dim = 1)
        
        full_events = torch.concat([torch.zeros(list(delta_times.shape)+[self.__sim_size]), events[:,1:].unsqueeze(2)], dim=2)
        full_events = full_events.flatten(1)
        bos_full_events = torch.concat([events[:,0].unsqueeze(1), full_events], dim = 1)
        return bos_full_times, bos_full_events.long()
    
    def __add_sim_times_no_batch(self, times, events):
        """
        Takes one sequence of times and events and adds sim_size auxiliar times
        
        args:
            times  - torch.Tensor of shape (max_len), that represents event arrival time since start
            events - torch.Tensor of shape (max_len), that represents event types
            
        returns:
            bos_full_times  - torch.Tensor of shape((sim_size + 1) * (max_len - 1) + 1) that consists of times and sim_size auxiliar times between events
            bos_full_events - torch.Tensor of shape((sim_size + 1) * (max_len - 1) + 1) that consists of events and sim_size zeros between events
        """
        delta_times = times[1:] - times[:-1]
        sim_delta_times = (torch.rand(list(delta_times.shape)+[self.__sim_size]) * delta_times.unsqueeze(1)).sort(dim=1).values
        full_times = torch.concat([sim_delta_times,delta_times.unsqueeze(1)], dim = 1)
        full_times = full_times + times[:-1].unsqueeze(1)
        full_times[delta_times<0,:] = 0
        full_times = full_times.flatten(0)
        bos_full_times = torch.concat([torch.zeros(1), full_times], dim = 0)
        
        full_events = torch.concat([torch.zeros(list(delta_times.shape)+[self.__sim_size]), events[1:].unsqueeze(1)], dim=1)
        full_events = full_events.flatten(0)
        bos_full_events = torch.concat([events[0].unsqueeze(0), full_events], dim = 0)
        return bos_full_times, bos_full_events.long()
    
    def __getitem__(self, idx):
        """
        Returns times + sim_times, events, lengths and true_ids of ids with true_times and true_events
        
        args:
            idx - Iterable of ids, slice or one id to return
            
        returns:
            times - output of __add_sim_times[_no_batch]
            events - output of __add_sim_time[_no_batch]
            lengths - torch.Tensor of sequence lengths
            true_ids_mask - torch.Tensor of true ids labeled with True
        """
        if isinstance(idx, Iterable) or isinstance(idx, slice):
            times, events = self.__add_sim_times(self.__times[idx], self.__events[idx])
            true_ids_mask = torch.zeros_like(times)
            for i in range(times.shape[0]):
                true_ids_mask[i,self.__true_ids[:self.__lengths[i]]] = 1
            true_ids_mask = true_ids_mask.bool()
        else:
            times, events = self.__add_sim_times_no_batch(self.__times[idx], self.__events[idx])
            true_ids_mask = torch.zeros_like(times)
            true_ids_mask[self.__true_ids[:self.__lengths[idx]]] = 1
            true_ids_mask = true_ids_mask.bool()
        return times, events, self.__lengths[idx], true_ids_mask
    
    def __len__(self):
        """
        Returns the number of datapoints
        """
        return self.__times.shape[0]
