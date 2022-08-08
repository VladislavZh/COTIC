import torch
from torch.utils.data import Dataset

from typing import List, Tuple

class EventData(Dataset):
    """
    Base event sequence dataset, takes list of sequences of event times and event types, pads them and returns event_time torch Tensor and event type torch Tensor
    """
    def __init__(
        self,
        times: List[torch.Tensor],
        events: List[torch.Tensor]
    ):
        self.__times, self.__events = self.__pad(times, events)
        
    @staticmethod
    def __pad(times, events):
        """
        Takes list of times and events sequences, pads them
        
        args:
            times   - list of torch.Tensor of shape = (length,), that represents event arrival time since start
            events  - list of torch.Tensor of shape = (length,), that represents event types in {0,1,...,C-1}
        
        returns:
            times   - torch.Tensor of shape (dset_size, max_len), where max_len = max({length}) + 1, padded times
            events  - torch.Tensor of shape (dset_size, max_len), where max_len = max({length}) + 1, padded events with shifted events
                                                                 {0,...,C-1} -> {1,...,C}, 0 as pad value
        """
        lengths = torch.Tensor([len(time_seq) for time_seq in times]).long()
        max_len = torch.max(lengths)
        
        tensor_times, tensor_events = torch.zeros(len(times), max_len), torch.zeros(len(times), max_len)
        
        for i, l in enumerate(lengths):
            tensor_times[i,:l] = times[i]
            tensor_events[i,:l] = events[i] + 1
            
        dt = tensor_times[:,1:] - tensor_times[:,:-1]
        dt = dt[dt>0]
        dt_median = torch.median(dt)
        print('Max dt =', torch.max(dt)/dt_median)
        
        return tensor_times/dt_median, tensor_events.long()
    
    def __len__(self):
        return len(self.__times)
    
    def __getitem__(self, idx):
        return self.__times[idx], self.__events[idx]
