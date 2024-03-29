import os
import pickle
import re
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import torch
import tqdm
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


class Data_preprocessor:
    def __init__(self):
        self.le = LabelEncoder()
        self.le.fit([])
        self.min_max = MinMaxScaler()

    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        for key in data.keys():
            if key not in ["time", "event"]:
                data.drop(key, axis=1, inplace=True)

        # need these only for stream label encoding
        if set(np.unique(data["event"])).issubset(self.le.classes_):
            # just transform
            data["event"] = np.squeeze(
                self.le.transform(data["event"].values.reshape(-1, 1))
            )
        else:
            # add new labels to the end of classes array and transform
            self.le.classes_ = np.array(
                list(self.le.classes_)
                + list(set(np.unique(data["event"])) - set(self.le.classes_))
            )
            data["event"] = np.squeeze(
                self.le.transform(data["event"].values.reshape(-1, 1))
            )

        try:
            data["time"] = np.squeeze(
                self.min_max.transform(data["time"].values.reshape(-1, 1))
            )
        except NotFittedError:
            data["time"] = np.squeeze(
                self.min_max.fit_transform(data["time"].values.reshape(-1, 1))
            )

        return data


def load_data(
    data_dir: str,
    unix_time: bool,
    dataset_size: Optional[int],
    max_len: Optional[int],
    preprocess_type: str = "default",
) -> List[torch.Tensor]:
    times = []
    events = []
    if preprocess_type == "default":
        data_preprocessor = Data_preprocessor()
    count = 0
    for f in tqdm.tqdm(
        sorted(
            os.listdir(data_dir),
            key=lambda x: (
                int(re.sub(rf".csv", "", x)) if re.sub(rf".csv", "", x).isdigit() else 0
            ),
        )
    ):
        if f.endswith(f".csv") and re.sub(rf".csv", "", f).isnumeric():
            df = pd.read_csv(data_dir + "/" + f)
            df = df.sort_values(by=["time"])
            if preprocess_type == "default":
                df = data_preprocessor.prepare_data(df)
            t = torch.Tensor(list(df["time"]))
            e = torch.Tensor(list(df["event"]))
            if max_len is not None:
                t = t[:max_len]
                e = e[:max_len]
            times.append(t)
            events.append(e)
            if unix_time:
                times[-1] /= 86400
            count += 1
            if dataset_size is not None:
                if count == dataset_size:
                    break
    return times, events, data_preprocessor if preprocess_type is not None else None


def load_data_simple(
    data_dir: str,
    stage: str,
    dataset_size: int,
    data_type: str,
    max_len: Optional[int],
    event_types: Optional[Union[int, torch.Tensor]],
) -> List[torch.Tensor]:
    times = []
    events = []
    cur = 0
    if data_type == ".csv":
        data_dir = os.path.join(data_dir, stage)
        for f in tqdm.tqdm(
            sorted(
                os.listdir(data_dir),
                key=lambda x: (
                    int(re.sub(rf".csv", "", x))
                    if re.sub(rf".csv", "", x).isdigit()
                    else 0
                ),
            )
        ):
            if f.endswith(f".csv") and re.sub(rf".csv", "", f).isnumeric():
                df = pd.read_csv(data_dir + "/" + f)
                df = df.sort_values(by=["time"])
                t = torch.Tensor(list(df["time"]))
                e = torch.Tensor(list(df["event"]))
                if max_len is not None:
                    t = t[:max_len]
                    e = e[:max_len]
                times.append(t)
                events.append(e)
                cur += 1
                if cur == dataset_size:
                    break
    elif data_type == ".pkl":
        # in pkl files dev stands for val
        stage = stage.replace("val", "dev")
        pkl_file_path = os.path.join(data_dir, stage + ".pkl")
        with open(pkl_file_path, "rb") as pkl_file:
            pkl = pickle.load(pkl_file)

        for seq in tqdm.tqdm(pkl[stage]):
            df = pd.DataFrame.from_records(seq)
            # columns are "time_since_start", "time_since_last_event", "type_event"
            df = df.sort_values(by=["time_since_start"])
            t = torch.Tensor(list(df["time_since_start"]))
            e = torch.Tensor(list(df["type_event"]))
            if max_len is not None:
                t = t[:max_len]
                e = e[:max_len]
            times.append(t)
            events.append(e)
            cur += 1
            if cur == dataset_size:
                break
    else:
        raise SystemExit(f"Error: data type {data_type} is not supported")

    unique_events = event_types
    final_times = times
    final_events = events

    if event_types is not None:
        if isinstance(event_types, int):
            all_events = torch.concat(events)
            unique_events = torch.unique(all_events)
            num_events = torch.Tensor(
                [torch.sum(all_events == event) for event in unique_events]
            )
            _, ids = torch.sort(num_events, descending=True)
            unique_events = unique_events[ids][:event_types]
        else:
            unique_events = event_types
        for i in range(len(events)):
            mask = torch.isin(events[i], unique_events)
            times[i] = times[i][mask]
            events[i] = events[i][mask]
        final_times = []
        final_events = []
        for i in range(len(events)):
            if len(times[i]) > 1:
                final_times.append(times[i])
                final_events.append(events[i])

        print(len(final_times))

    return final_times, final_events, unique_events
