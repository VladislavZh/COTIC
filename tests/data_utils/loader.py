import unittest
import os
import pandas as pd
import torch
from src.utils.data_utils.loader import load_time_series_data


class TestLoadTimeSeriesData(unittest.TestCase):
    """Test suite for load_time_series_data function."""

    def setUp(self):
        """Set up the test environment."""
        # Create a temporary directory and generate some test CSV files for testing
        self.test_dir = "test_data"
        os.makedirs(self.test_dir, exist_ok=True)

        # Create test CSV files with time and event columns
        data1 = {'time': [1, 2, 3], 'event': [0.1, 0.2, 0.3]}
        data2 = {'time': [4, 5, 6], 'event': [0.4, 0.5, 0.6]}
        pd.DataFrame(data1).to_csv(os.path.join(self.test_dir, "1.csv"), index=False)
        pd.DataFrame(data2).to_csv(os.path.join(self.test_dir, "2.csv"), index=False)

        # Create additional non-CSV files
        open(os.path.join(self.test_dir, "redundant.txt"), 'w').close()
        open(os.path.join(self.test_dir, "dump.csv"), 'w').close()

    def test_load_time_series_data(self):
        """Test loading time series data_utils."""
        # Test loading data_utils without specifying dataset size
        time_series, events_series = load_time_series_data(self.test_dir)
        self.assertEqual(len(time_series), 2)  # Asserts that two files were loaded
        self.assertIsInstance(time_series[0], torch.Tensor)  # Asserts that the data_utils is a Torch tensor

        # Test loading data_utils with a specified dataset size of 1
        time_series_limited, events_series_limited = load_time_series_data(self.test_dir, dataset_size=1)
        self.assertEqual(len(time_series_limited), 1)  # Asserts that only one file was loaded

    def tearDown(self):
        """Tear down the test environment."""
        # Remove the temporary test directory and its contents
        if os.path.exists(self.test_dir):
            for file_name in os.listdir(self.test_dir):
                file_path = os.path.join(self.test_dir, file_name)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            os.rmdir(self.test_dir)


if __name__ == '__main__':
    unittest.main()
