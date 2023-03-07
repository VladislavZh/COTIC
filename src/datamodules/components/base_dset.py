import torch
from torch.utils.data import Dataset

from typing import List, Tuple


class EventData(Dataset):
    """Base event sequence dataset: Takes list of sequences of event times and event types and pads."""

    def __init__(self, times: List[torch.Tensor], events: List[torch.Tensor]) -> None:
        self.__times, self.__events = self.__pad(times, events)

    @staticmethod
    def __pad(
        times: List[torch.Tensor], events: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Take lists of times and events sequences, pads them with zeros.

        :param times: list of torch.Tensor of shape = (length,), that represents event arrival time since start
        :param events: list of torch.Tensor of shape = (length,), that represents event types in {0, 1, ..., C-1}

        :returns: tuple of
            - times: torch.Tensor of shape (dset_size, max_len), where max_len = max({length}) + 1, padded times
            - events: torch.Tensor of shape (dset_size, max_len), where max_len = max({length}) + 1, padded events with shifted events
                                                                 {0, ..., C-1} -> {1, ..., C}, 0 as pad value
        """
        lengths = torch.Tensor([len(time_seq) for time_seq in times]).long()
        max_len = torch.max(lengths)

        tensor_times = torch.zeros(len(times), max_len)
        tensor_events = torch.zeros(len(events), max_len)

        dts = torch.empty(1)[1:]

        for i, l in enumerate(lengths):
            tensor_times[i, :l] = times[i]

            if len(times[i]) != len(events[i]):
                continue

            dts = torch.concat([dts, times[i][1:] - times[i][:-1]])

            tensor_events[i, :l] = events[i] + 1

        return tensor_times, tensor_events.long()

    def __len__(self) -> int:
        return len(self.__times)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.__times[idx], self.__events[idx]
