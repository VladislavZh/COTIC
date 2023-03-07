from typing import List, Optional, Any, Tuple

import tqdm
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder  # , StandardScaler
from sklearn.exceptions import NotFittedError


class MinMaxScaler(torch.nn.Module):
    """Re-implementation of Min-Max scaler."""

    def __init__(self) -> None:
        self.min = None
        self.max = None

    def fit(self, array: torch.Tensor) -> None:
        """Fit scaler: find min and max values.

        :param array: data to be scaled
        """
        self.min = torch.amin(array)
        self.max = torch.amax(array)

    def transform(self, array: torch.Tensor) -> torch.Tensor:
        """Apply transformation.

        :param array: data to be scaled

        :return: scaled data
        """
        if self.max is None or self.min is None:
            raise NotFittedError
        scaled = (array - self.min) / (self.max - self.min)
        return scaled

    def fit_transform(self, array: torch.Tensor) -> torch.Tensor:
        """Fit scaler and apply transform.

        :param array: data to be scaled

        :return: scaled data
        """
        self.min = torch.amin(array)
        self.max = torch.amax(array)
        scaled = (array - self.min) / (self.max - self.min)
        return scaled

    def inverse_transform(self, array: torch.Tensor) -> torch.Tensor:
        """Apply inverse transform.

        :param array: scaled data

        :return: descaled data
        """
        descaled = array * (self.max - self.min) + self.min
        return descaled


class StandardScaler(torch.nn.Module):
    """Re-implementation of Standard Scaler."""

    def __init__(self) -> None:
        self.mean = None
        self.std = None

    def fit(self, array: torch.Tensor) -> None:
        """Fit scaler: find min and max values.

        :param array: data to be scaled
        """
        self.mean = torch.mean(array)
        self.std = torch.std(array)

    def transform(self, array: torch.Tensor) -> torch.Tensor:
        """Apply transformation.

        :param array: data to be scaled

        :return: scaled data
        """
        if self.mean is None or self.std is None:
            raise NotFittedError
        # add small constant to avoid division by zero
        EPS = 1e-32
        scaled = (array - self.mean) / (self.std + EPS)
        return scaled

    def fit_transform(self, array: torch.Tensor) -> torch.Tensor:
        """Fit scaler and apply transform.

        :param array: data to be scaled

        :return: scaled data
        """
        self.mean = torch.mean(array)
        self.std = torch.std(array)
        # use small constant for denormalization as well
        EPS = 1e-32
        scaled = (array - self.mean) / (self.std + EPS)
        return scaled

    def inverse_transform(self, array: torch.Tensor) -> torch.Tensor:
        """Apply inverse transform.

        :param array: scaled data

        :return: descaled data
        """
        EPS = 1e-32
        descaled = array * (self.std + EPS) + self.mean
        # descaled = array * (self.max - self.min) + self.min
        return descaled


def get_scaler(name_scaler: str) -> Optional[Any]:
    """Initializer of the scaler.

    :param name_scaler: name of the scaler

    :return: scaler (Standard or MixMax), or None if no scaler should be used
    """
    scalers = {
        "standard_scaler": StandardScaler(),
        "min_max_scaler": MinMaxScaler(),
        "default": None,
    }
    return scalers[name_scaler]


class Data_preprocessor:
    """Class for data preprocessing."""

    def __init__(self):
        self.le = LabelEncoder()
        self.le.fit([])

    def prepare_data(
        self,
        times: np.array,
        events: np.array,
        scale_name: str,
        number_max: float,
        number_min: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Preprocess data: encode classes, transform times.

        :param times: sequence of times
        :param events: sequence of event types
        :param scale_name: name of the time scaler
        :param number_max: maximum time moment
        :param number_min: minimum time moment

        :return: tuple of
            - preprocessed times
            - preprocessed event types
        """
        # need these only for stream label encoding
        """
        if set(np.unique(events)).issubset(self.le.classes_):
            # just transform
            events = np.squeeze(self.le.transform(events.to_numpy().reshape(-1, 1)))
        else:
            # add new labels to the end of classes array and transform
            self.le.classes_ = np.array(
                list(self.le.classes_)
                + list(set(np.unique(events)) - set(self.le.classes_))
            )
            events = np.squeeze(self.le.transform(events.to_numpy().reshape(-1, 1)))
        """
        self.scaler = get_scaler(scale_name)
        self.scaler_name = scale_name
        times = self.normalization(times, number_max, number_min)

        return times, events

    def normalization(
        self, time: torch.Tensor, number_max: float, number_min: float
    ) -> torch.Tensor:
        """Scale times sequence.

        :param time: raw times sequences
        :param number_max: maximum time moment for scaling
        :param number_min: minimum time moment for scaling

        :return: scaled times sequence
        """
        if self.scaler is not None:
            try:
                time = torch.squeeze(
                    self.scaler.transform(torch.Tensor(time.values.reshape(-1, 1)))
                )
            except NotFittedError:
                self.scaler.fit(torch.Tensor([number_max, number_min]).reshape(-1, 1))
                time = torch.squeeze(
                    self.scaler.fit_transform(torch.Tensor(time.values.reshape(-1, 1)))
                )

        return time

    def denormalization(self, output: torch.Tensor) -> torch.Tensor:
        """Apply inverse scaling transform and denormilize output.

        :param output: scaled time sequence

        :return: descaled times sequence
        """
        if self.scaler is not None:
            output = self.scaler.inverse_transform(output)

        return output


def load_data_parquet(
    data_path: str,
    time_col: str = "time",
    event_col: str = "event",
    seq_col: str = "sequence_id",
    unix_time_flag: bool = False,
    preprocess_type: str = None,
    num_types: int = None,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Reads parquet file with data of all sequences, and separates them into times / events.

    :param data_path: path to .parquet file with data
    :param time_col: name of the column with times
    :param event_col: name of the column with event types
    :param seq_col: name of the column with sequence id
    :param unix_time_flag: if True, divide all times by 86400
    :param preprocess_type: type of the time preprocessing (scaler name)

    :return: tuple of
        - list of times sequences
        - list of event types sequences
    """

    # data path example is /home/.../experiemnt_name.parquet
    experiment_name = data_path.split("/")[-1].split(".")[0]

    assert unix_time_flag == (
        experiment_name.lower() in ["amazon", "atm", "retweet"]
    ), "unix_time flag should be used for 'Amazon', 'Retweet' and 'ATM' datasets only."

    times = []
    events = []
    full_data = pd.read_parquet(data_path, engine="pyarrow")
    sequences = [x.reset_index(drop=True) for _, x in full_data.groupby([seq_col])]

    # "default" preprocess_type is for no preprocessing
    # if preprocess_type != "default":
    data_preprocessor = Data_preprocessor()

    all_times_list = []
    for seq in tqdm.tqdm(sequences):
        all_times_list.extend(list(seq[time_col]))

    number_max = np.max(all_times_list)
    number_min = np.min(all_times_list)
    number_quantile_95 = np.quantile(all_times_list, 0.95)
    number_quantile_05 = np.quantile(all_times_list, 0.05)

    max_event_num_list = []
    for seq in tqdm.tqdm(sequences):
        time_seq = seq[time_col]
        event_seq = seq[event_col]

        max_event_num_list.append(max(event_seq))

        if unix_time_flag:
            time_seq /= 86400

        # "default" preprocess_type is for no preprocessing
        # if preprocess_type != "default":
        time_seq, event_seq = data_preprocessor.prepare_data(
            time_seq,
            event_seq,
            preprocess_type,
            # use quantiles instead of minimum and maximum values for MinMax Scaler
            number_quantile_95,
            number_quantile_05,
        )
        times.append(torch.Tensor(time_seq))
        events.append(torch.Tensor(event_seq))

    event_types_num = max(max_event_num_list) + 1

    print("event_types_num:", event_types_num)
    print("num types in config:", num_types)

    assert (
        event_types_num == num_types
    ), "Number of event types in the config file is not consistent with the dataset."

    return times, events, data_preprocessor
