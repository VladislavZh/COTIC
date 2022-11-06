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
        self.__times, self.__events, self.__times_targets, self.__events_targets = self.__pad(times, events)
        
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
        max_len = torch.max(lengths) - 1
        
        tensor_times, tensor_events = torch.zeros(len(times), max_len), torch.zeros(len(times), max_len)
        tensor_times_targets = torch.zeros(len(times))
        tensor_events_targets = torch.zeros(len(times))
        for i, l in enumerate(lengths):
            tensor_times[i,:l-1] = times[i][:l-1]
            tensor_times_targets[i] = times[i][-1]
            tensor_events[i,:l-1] = events[i][:l-1] + 1
            tensor_events_targets[i] = events[i][-1] + 1

        return tensor_times, tensor_events.long(), tensor_times_targets, tensor_events_targets.long()
    
    def __len__(self):
        return len(self.__times)
    
    def __getitem__(self, idx):
        return self.__times[idx], self.__events[idx], self.__times_targets[idx], self.__events_targets[idx]
