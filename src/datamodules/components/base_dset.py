import torch
from torch.utils.data import Dataset


class EventDataset(Dataset):
    """
    Event sequence dataset.
    This dataset takes a list of sequences of event times and event types,
    pads them, and returns event time and event type torch Tensors.
    """

    def __init__(self, event_times: list[torch.Tensor], event_types: list[torch.Tensor], num_event_types: int):
        """
        Initializes the EventDataset.

        Args:
        - event_times (list[torch.Tensor]): List of torch.Tensor of shape=(length,),
                                            representing event arrival times since the start.
        - event_types (list[torch.Tensor]): List of torch.Tensor of shape=(length,),
                                            representing event types in {0, 1, ..., C-1}.
        - num_event_types (int): Number of unique event types.
        """
        self.num_event_types = num_event_types
        self.__event_times, self.__event_types = self.__add_bos(*self.__pad(event_times, event_types))

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

    def __add_bos(self, event_times: torch.Tensor, event_types: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Add beginning of sequence (BOS) tokens to event times and event types.

        Args:
        - event_times (torch.Tensor): Tensor of event times.
        - event_types (torch.Tensor): Tensor of event types.

        Returns:
        - Tuple[torch.Tensor, torch.Tensor]: Tuple containing event times and event types with BOS tokens added.
        """
        bos_event_times = torch.cat([torch.zeros(event_times.shape[0], 1), event_times], dim=1)
        bos_event_types = torch.cat([torch.full((event_types.shape[0], 1), self.num_event_types + 1, dtype=torch.long), event_types], dim=1)
        return bos_event_times, bos_event_types

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
        return event_times, event_types
