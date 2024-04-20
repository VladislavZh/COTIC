from typing import Type, Union, Optional

import torch
import random
from torch.utils.data import Dataset

from src.utils.data_utils.normalizers import Normalizer


class EventDataset(Dataset):
    """
    Event sequence dataset.
    This dataset takes a list of sequences of event times and event types,
    pads them, and returns event time and event type torch Tensors.
    """

    def __init__(
            self,
            event_times: list[torch.Tensor],
            event_types: list[torch.Tensor],
            num_event_types: int,
            train_random_crop: bool
    ):
        """
        Initializes the EventDataset.

        Args:
        - event_times (list[torch.Tensor]): List of torch.Tensor of shape=(length,),
                                            representing event arrival times since the start.
        - event_types (list[torch.Tensor]): List of torch.Tensor of shape=(length,),
                                            representing event types in {0, 1, ..., C-1}.
        - num_event_types (int): Number of unique event types.
        - crop_size (int | None): Crop size value.
        """
        self.num_event_types = num_event_types
        self.train_random_crop = train_random_crop
        self.__event_times, self.__event_types = self.__pad(event_times, event_types)

    def normalize_data(self, normalizer: Union[Type[Normalizer], Normalizer]) -> Normalizer:
        """
        Normalizes the inter-event times of the dataset using the provided Normalizer.

        This method applies a normalization transformation to the inter-event times of the dataset.
        If a Normalizer class type is provided, it initializes an instance using the non-padded
        inter-event times data. The normalization is then applied to all event times in the dataset.

        Args:
        - normalizer (Union[Type[Normalizer], Normalizer]): The Normalizer class or an instance of a Normalizer.
                                                            If a class type is provided, it must be a subclass
                                                            of Normalizer and will be instantiated using the
                                                            data from this dataset.

        Returns:
        - Normalizer: The normalizer instance used to normalize the dataset's inter-event times.
                      If a Normalizer instance was provided, it is returned directly.
                      If a Normalizer class type was provided, the newly instantiated Normalizer is returned.
        """
        if isinstance(normalizer, type) and issubclass(normalizer, Normalizer):
            delta_times = self.__event_times[:, 1:] - self.__event_times[:, :-1]
            non_pad_mask = self.__event_types.ne(0)[:, 1:]

            normalizer = normalizer.from_data(delta_times[non_pad_mask])
        self.__event_times = normalizer.normalize(self.__event_times)
        return normalizer

    @staticmethod
    def __pad(event_times: list[torch.Tensor], event_types: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Pads a list of event times and event types sequences.

        Args:
        - event_times (list[torch.Tensor]): List of torch.Tensor of shape=(length,),
                                            representing event arrival times since the start.
        - event_types (list[torch.Tensor]): List of torch.Tensor of shape=(length,),
                                            representing event types in {0, 1, ..., C-1}.

        Returns:
        - tensor_event_times (torch.Tensor): Torch tensor of shape=(dataset_size, max_len),
                                             where max_len = max({length}) + 1, padded event times.
        - tensor_event_types (torch.Tensor): Torch tensor of shape=(dataset_size, max_len),
                                              where max_len = max({length}) + 1,
                                              padded event types with shifted events
                                              {0, ..., C-1} -> {1, ..., C}, 0 as the pad value.
        """
        lengths = torch.LongTensor([time_seq.size(0) for time_seq in event_times])
        max_len = int(torch.max(lengths))

        tensor_event_times = torch.zeros(len(event_times), max_len)
        tensor_event_types = torch.zeros(len(event_times), max_len)

        for i, (time_seq, event_seq) in enumerate(zip(event_times, event_types)):
            seq_len = time_seq.size(0)
            tensor_event_times[i, :seq_len] = time_seq
            tensor_event_types[i, :seq_len] = event_seq + 1

        return tensor_event_times, tensor_event_types.long()

    def __len__(self) -> int:
        """Returns the length of the dataset."""
        return len(self.__event_times)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves an item from the dataset at the specified index.

        Args:
        - idx (int): Index of the item to retrieve.

        Returns:
        - Tuple[torch.Tensor, torch.Tensor]: Tuple containing event times and event types at the given index.
        """
        event_times, event_types = self.__event_times[idx], self.__event_types[idx]

        if self.train_random_crop:
            max_idx = int(torch.sum(event_types.ne(0)))
            begin_idx = random.randint(0, max_idx - 1)
            end_idx = random.randint(begin_idx + 1, max_idx)
            event_times[:end_idx - begin_idx] = event_times[begin_idx:end_idx].clone()
            event_times[end_idx - begin_idx:] = 0
            event_types[:end_idx - begin_idx] = event_types[begin_idx:end_idx].clone()
            event_types[end_idx - begin_idx:] = 0

        return event_times, event_types
