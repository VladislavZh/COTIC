import os
import re
from typing import Optional, Tuple, List
import pandas as pd
import torch
from tqdm import tqdm


def load_time_series_data(
        data_directory: str,
        dataset_size: Optional[int] = None
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Load time series data from CSV files in a directory.

    Args:
    - data_directory (str): Path to the directory containing CSV files.
    - dataset_size (Optional[int]): Optional size limit for the dataset. If specified,
      only the first `dataset_size` files will be read.

    Returns:
    - Tuple[List[torch.Tensor], List[torch.Tensor]]: A tuple containing two lists:
      1. List of Torch tensors representing time series data.
      2. List of Torch tensors representing corresponding events.
    """

    time_series = []
    events_series = []

    # Filter CSV files in the directory based on numeric filenames
    csv_files = sorted(
        filter(
            lambda file: re.match(r"\d+\.csv$", file),
            os.listdir(data_directory)
        ),
        key=lambda x: int(re.sub(r"\D", "", x)) if re.sub(r"\D", "", x).isdigit() else 0
    )

    # Read data from CSV files
    for count, file_name in tqdm(enumerate(csv_files), desc="Loading Data"):
        if dataset_size is not None and count == dataset_size:
            break

        file_path = os.path.join(data_directory, file_name)
        df = pd.read_csv(file_path)
        df = df.sort_values(by=["time"])

        time_values = torch.Tensor(df["time"].values)
        event_values = torch.Tensor(df["event"].values)

        time_series.append(time_values)
        events_series.append(event_values)

    return time_series, events_series
