import os
from typing import Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split

from src.utils.data_utils import load_data, load_data_simple

from .components.base_dset import EventData


class EventDataModule(LightningDataModule):
    """
    General event sequence datamodule
    """

    def __init__(
        self,
        data_dir: str = "data/",
        data_type: str = ".csv",
        unix_time: bool = False,
        train_val_test_split: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        batch_size: int = 64,
        dataset_size: Optional[int] = None,
        dataset_size_train: Optional[int] = None,
        dataset_size_val: Optional[int] = None,
        dataset_size_test: Optional[int] = None,
        max_len: Optional[int] = None,
        num_event_types: Optional[int] = None,
        num_workers: int = 0,
        pin_memory: bool = False,
        random_seed: int = 42,
        preprocess_type: str = "default",
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        if not self.data_train and not self.data_val and not self.data_test:
            if "preprocess_type" in self.hparams.keys():
                times, events = load_data(
                    self.hparams.data_dir,
                    self.hparams.unix_time,
                    self.hparams.dataset_size,
                    self.hparams.max_len,
                    self.hparams.preprocess_type,
                )
            else:
                times, events = load_data(
                    self.hparams.data_dir,
                    self.hparams.unix_time,
                    self.hparams.dataset_size,
                    self.hparams.max_len,
                )

            dataset = EventData(times, events)
            N = len(dataset)
            lengths = [int(N * v) for v in self.hparams.train_val_test_split]
            lengths[0] = N - (lengths[1] + lengths[2])
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset,
                lengths=lengths,
                generator=torch.Generator().manual_seed(self.hparams.random_seed),
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )


class EventDataModuleSplitted(EventDataModule):
    def setup(self, stage: Optional[str] = None):
        self.data_process = None
        if not self.data_train and not self.data_val and not self.data_test:
            times, events, unique_events = load_data_simple(
                self.hparams.data_dir,
                "train",
                self.hparams.dataset_size_train,
                self.hparams.data_type,
                self.hparams.max_len,
                self.hparams.num_event_types,
            )
            self.data_train = EventData(times, events)
            times, events, _ = load_data_simple(
                self.hparams.data_dir,
                "val",
                self.hparams.dataset_size_val,
                self.hparams.data_type,
                self.hparams.max_len,
                unique_events,
            )
            self.data_val = EventData(times, events)
            times, events, _ = load_data_simple(
                self.hparams.data_dir,
                "test",
                self.hparams.dataset_size_test,
                self.hparams.data_type,
                self.hparams.max_len,
                unique_events,
            )
            self.data_test = EventData(times, events)
