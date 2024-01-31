from typing import Optional
import os

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from .components.base_dset import EventDataset

from src.utils.data_utils import load_time_series_data


class EventDataModule(LightningDataModule):
    """
    General event sequence data module.
    This LightningDataModule handles loading and preparing event sequence datasets
    for training, validation, and testing in PyTorch Lightning workflows.
    """

    def __init__(
            self,
            num_event_types: int,
            data_dir: str = "data/",
            batch_size_train: int = 20,
            batch_size_val_test: int = 1,
            dataset_size_train: Optional[int] = None,
            dataset_size_val: Optional[int] = None,
            dataset_size_test: Optional[int] = None,
            num_workers: int = 0,
            pin_memory: bool = False,
    ):
        """
        Initializes the EventDataModule.

        Args:
            num_event_types (int): Number of unique event types.
            data_dir (str): Path to the directory containing the dataset.
            batch_size_train (int): Batch size for training data loading.
            batch_size_val_test (int): Batch size for validation and test data loading.
            dataset_size_train (Optional[int]): Size of the training dataset.
            dataset_size_val (Optional[int]): Size of the validation dataset.
            dataset_size_test (Optional[int]): Size of the test dataset.
            num_workers (int): Number of workers for data loading.
            pin_memory (bool): Flag to enable memory pinning.
        """
        super().__init__()

        self.num_event_types = num_event_types
        self.data_dir = data_dir
        self.batch_size_train = batch_size_train
        self.batch_size_val_test = batch_size_val_test
        self.dataset_size_train = dataset_size_train
        self.dataset_size_val = dataset_size_val
        self.dataset_size_test = dataset_size_test
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.data_train = None
        self.data_val = None
        self.data_test = None

    def load_event_data(self, data_path: str, dataset_size: Optional[int]) -> EventDataset:
        """
        Loads event data from a specified path and creates an EventData instance.

        Args:
            data_path (str): Path to the dataset.
            dataset_size (Optional[int]): Size of the dataset.

        Returns:
            EventData: An instance of EventData with loaded data.
        """
        times, events = load_time_series_data(data_path, dataset_size)
        return EventDataset(times, events, self.num_event_types)

    def setup(self, stage: Optional[str] = None):
        """
        Sets up the datasets for training, validation, and testing.

        Args:
            stage (Optional[str]): Stage of setup.
        """
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = self.load_event_data(
                os.path.join(self.data_dir, "train"), self.dataset_size_train
            )
            self.data_val = self.load_event_data(
                os.path.join(self.data_dir, "val"), self.dataset_size_val
            )
            self.data_test = self.load_event_data(
                os.path.join(self.data_dir, "test"), self.dataset_size_test
            )

    def create_dataloader(self, dataset, batch_size: int, shuffle: bool = False) -> DataLoader:
        """
        Creates a DataLoader instance for the given dataset with specified batch size.

        Args:
            dataset: Dataset to create DataLoader for.
            batch_size (int): Batch size for the DataLoader.
            shuffle (bool): Whether to shuffle the data.

        Returns:
            DataLoader: DataLoader instance for the dataset.
        """
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=shuffle,
        )

    def train_dataloader(self) -> DataLoader:
        """Returns a DataLoader for the training dataset."""
        return self.create_dataloader(self.data_train, self.batch_size_train, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """Returns a DataLoader for the validation dataset."""
        return self.create_dataloader(self.data_val, self.batch_size_val_test)

    def test_dataloader(self) -> DataLoader:
        """Returns a DataLoader for the test dataset."""
        return self.create_dataloader(self.data_test, self.batch_size_val_test)
