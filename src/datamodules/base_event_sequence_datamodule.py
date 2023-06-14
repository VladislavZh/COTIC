from pathlib import Path
from typing import Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from .components.base_dset import EventData

from src import utils
from src.utils.data_utils import load_data

log = utils.get_logger(__name__)

class EventDataModule(LightningDataModule):
    """
    General event sequence datamodule
    """

    def __init__(
        self,
        data_dir: str = "data/",
        unix_time: bool = False,
        train_val_test_split: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        batch_size: int = 64,
        dataset_size_train: Optional[int] = None,
        dataset_size_val: Optional[int] = None,
        dataset_size_test: Optional[int] = None,
        num_workers: int = 0,
        pin_memory: bool = False,
        random_seed: int = 42,
        preprocess_type: str = None
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
            #if "preprocess_type" in self.hparams.keys():
            # Train dataset
            train_times, train_events, self.data_process_train = load_data(Path(self.hparams.data_dir) / "train", self.hparams.unix_time,
                                      self.hparams.dataset_size_train, self.hparams.preprocess_type)
            self.data_train = EventData(train_times, train_events)

            # Val dataset
            val_times, val_events, self.data_process_val = load_data(Path(self.hparams.data_dir) / "val", self.hparams.unix_time,
                                      self.hparams.dataset_size_val, self.hparams.preprocess_type)
            self.data_val = EventData(val_times, val_events)

            # Test dataset
            test_times, test_events, self.data_process_test = load_data(Path(self.hparams.data_dir) / "test",
                                                                        self.hparams.unix_time,
                                                                        self.hparams.dataset_size_test,
                                                                        self.hparams.preprocess_type)
            self.data_test = EventData(test_times, test_events)

    # def setup(self, stage: Optional[str] = None):
    #     if not self.data_train and not self.data_val and not self.data_test:
    #         if "preprocess_type" in self.hparams.keys():
    #             times, events, self.data_process = load_data(self.hparams.data_dir, self.hparams.unix_time, self.hparams.dataset_size, self.hparams.preprocess_type)
    #         else:
    #             times, events, self.data_process = load_data(self.hparams.data_dir, self.hparams.unix_time, self.hparams.dataset_size)
    #         dataset = EventData(times, events)
    #         N = len(dataset)
    #         lengths = [int(N * v) for v in self.hparams.train_val_test_split]
    #         lengths[0] = N - (lengths[1] + lengths[2])
    #         self.data_train, self.data_val, self.data_test = random_split(
    #             dataset=dataset,
    #             lengths=lengths,
    #             generator=torch.Generator().manual_seed(self.hparams.random_seed),
    #         )

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
