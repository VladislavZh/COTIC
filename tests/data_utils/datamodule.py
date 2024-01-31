import unittest
import os
import pandas as pd
import shutil
from src.datamodules.base_event_sequence_datamodule import EventDataModule


class TestEventDataModule(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory and generate some test CSV files for testing
        self.test_dir = "test_data"
        os.makedirs(os.path.join(self.test_dir, 'train'), exist_ok=True)
        os.makedirs(os.path.join(self.test_dir, 'val'), exist_ok=True)
        os.makedirs(os.path.join(self.test_dir, 'test'), exist_ok=True)

        # Create test CSV files with time and event columns for train, val, and test
        data1 = {'time': [1, 2, 3], 'event': [0, 1, 0]}
        data2 = {'time': [4, 5], 'event': [2, 0]}
        data3 = {'time': [6, 7, 8], 'event': [1, 2, 0]}
        pd.DataFrame(data1).to_csv(os.path.join(self.test_dir, "train/0.csv"), index=False)
        pd.DataFrame(data2).to_csv(os.path.join(self.test_dir, "val/0.csv"), index=False)
        pd.DataFrame(data3).to_csv(os.path.join(self.test_dir, "test/0.csv"), index=False)

        # Initialize the EventDataModule
        self.event_data_module = EventDataModule(
            num_event_types=3,
            data_dir=self.test_dir,
            batch_size_train=32,
            batch_size_val_test=64,
            dataset_size_train=3,
            dataset_size_val=2,
            dataset_size_test=3,
            num_workers=2,
            pin_memory=False
        )

    def test_dataloader_sizes(self):
        # Setup the data module
        self.event_data_module.setup()

        # Get the data loaders
        train_loader = self.event_data_module.train_dataloader()
        val_loader = self.event_data_module.val_dataloader()
        test_loader = self.event_data_module.test_dataloader()

        # Check if DataLoader batch sizes match the specified sizes
        self.assertEqual(train_loader.batch_size, 32)
        self.assertEqual(val_loader.batch_size, 64)
        self.assertEqual(test_loader.batch_size, 64)

        # Check if the number of batches matches the expected number based on the limited dataset sizes
        self.assertEqual(len(train_loader), 1)  # Expected only 1 batch for training (2 files with batch size 32)
        self.assertEqual(len(val_loader), 1)    # Expected only 1 batch for validation (1 file with batch size 64)
        self.assertEqual(len(test_loader), 1)   # Expected only 1 batch for testing (2 files with batch size 64)

    def tearDown(self):
        # Remove the temporary test directory and its contents
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)


if __name__ == '__main__':
    unittest.main()
