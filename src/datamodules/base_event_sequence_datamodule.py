from typing import Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from .components.base_dset import EventData

from src.utils.data_utils import load_data_parquet  # ,load_data


class EventDataModule(LightningDataModule):
    """General event sequence datamodule."""

    def __init__(
        self,
        data_dir: str = "data/",
        unix_time: bool = False,
        train_val_test_split: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        batch_size: int = 64,
        dataset_size: Optional[int] = None,
        num_workers: int = 0,
        pin_memory: bool = False,
        random_seed: int = 42,
        preprocess_type: str = None,
        num_types: int = None,
    ) -> None:
        """Initialize datamodule.

        :param data_dir: path to a directory with data
        :param unix_time: if True, divide all the timestamps in a dataset by 86400
        :param train_val_test_split: train-validation-test split ratio
        :param batch_size: batch size for the dataloaders
        :param dataset_size: size of the dataset (if not None, crop the dataset)
        :param num_workers: number of CPUs
        :param pin_memory: if True, the data loader will copy Tensors into CUDA pinned memory before returning them
        :param random_seed: random seed to be fixed
        :param preprocess_type: type of the preprocessing (time scaler) to be used ('default' no additional time scaling)
        :param num_types: number of event types in the dataset
        """
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self) -> None:
        """Data preparation is specified in src/utils/data_utils.py."""
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Load the dataset and split it into 3 parts: train, validation and test sets.

        :param stage: 'train', 'val' or 'test'
        """
        if not self.data_train and not self.data_val and not self.data_test:
            if "preprocess_type" in self.hparams.keys():
                # times, events, self.data_process = load_data(self.hparams.data_dir, self.hparams.unix_time, self.hparams.dataset_size, self.hparams.preprocess_type)

                # use datasets saved in .parquet files
                times, events, self.data_process = load_data_parquet(
                    data_path=self.hparams.data_dir,
                    unix_time_flag=self.hparams.unix_time,
                    # self.hparams.dataset_size,
                    preprocess_type=self.hparams.preprocess_type,
                    num_types=self.hparams.num_types,
                )
            # else:
            #    # times, events, self.data_process = load_data(self.hparams.data_dir, self.hparams.unix_time, self.hparams.dataset_size)
            #
            #    # use datasets saved in .parquet files
            #    times, events, self.data_process = load_data_parquet(
            #        data_path=self.hparams.data_dir,
            #        unix_time_flag=self.hparams.unix_time,
            #        num_types=self.hparams.num_types
            #        # self.hparams.dataset_size
            #    )
            dataset = EventData(times, events)
            N = len(dataset)
            lengths = [int(N * v) for v in self.hparams.train_val_test_split]
            lengths[0] = N - (lengths[1] + lengths[2])
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset,
                lengths=lengths,
                generator=torch.Generator().manual_seed(self.hparams.random_seed),
            )

    def train_dataloader(self) -> DataLoader:
        """Create train dataloader."""
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader."""
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        """Create test dataloader."""
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
