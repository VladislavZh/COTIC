import unittest
import torch
from src.datamodules.components.base_dset import EventDataset


class TestEventData(unittest.TestCase):
    def setUp(self):
        # Create sample event sequences for testing
        times = [
            torch.Tensor([1, 2, 3]),
            torch.Tensor([4, 5]),
            torch.Tensor([6, 7, 8, 9])
        ]
        events = [
            torch.Tensor([0, 1, 0]),
            torch.Tensor([1, 1]),
            torch.Tensor([2, 1, 0, 2])
        ]
        self.event_data = EventDataset(times, events, 3)

    def test_padding(self):
        # Test padding functionality
        padded_times, padded_events = self.event_data._EventDataset__pad(
            self.event_data._EventDataset__event_times,
            self.event_data._EventDataset__event_types
        )
        self.assertEqual(padded_times.shape, (3, 5))  # Asserts padded times shape
        self.assertEqual(padded_events.shape, (3, 5))  # Asserts padded events shape

    def test_dataset_length(self):
        # Test dataset length
        self.assertEqual(len(self.event_data), 3)  # Asserts the length of the dataset

    def test_get_item(self):
        # Test __getitem__ functionality
        event_time, event_type = self.event_data[1]
        self.assertTrue(torch.equal(event_time, torch.Tensor([0, 4, 5, 0, 0])))  # Asserts retrieved event times
        self.assertTrue(torch.equal(event_type, torch.Tensor([4, 2, 2, 0, 0])))  # Asserts retrieved event types


if __name__ == '__main__':
    unittest.main()
