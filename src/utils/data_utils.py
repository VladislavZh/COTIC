from typing import List, Optional
import os
import tqdm
import torch
import pandas as pd
import re
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.exceptions import NotFittedError


class Data_preprocessor():
    def __init__(self):
        self.le = LabelEncoder()
        self.le.fit([])
        self.min_max = MinMaxScaler()

    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        for key in data.keys():
            if key not in ['time', 'event']:
                data.drop(key, axis=1, inplace=True)

        # need these only for stream label encoding
        if set(np.unique(data['event'])).issubset(self.le.classes_):
            # just transform
            data['event'] = np.squeeze(self.le.transform(data['event'].values.reshape(-1, 1)))
        else:
            # add new labels to the end of classes array and transform
            self.le.classes_ = np.array(list(self.le.classes_)
                                        + list(set(np.unique(data['event']))
                                               - set(self.le.classes_)))
            data['event'] = np.squeeze(self.le.transform(data['event'].values.reshape(-1, 1)))

        try:
            data['time'] = np.squeeze(self.min_max.transform(data['time'].values.reshape(-1, 1)))
        except NotFittedError:
            data['time'] = np.squeeze(self.min_max.fit_transform(data['time'].values.reshape(-1, 1)))

        return data

def load_data(
    data_dir: str,
    unix_time: bool,
    dataset_size: Optional[int],
    max_len: Optional[int],
    preprocess_type: str = "default"
    ) -> List[torch.Tensor]:
    times = []
    events = []
    if preprocess_type == "default":
        data_preprocessor = Data_preprocessor()
    count = 0
    for f in tqdm.tqdm(sorted(
        os.listdir(data_dir),
        key=lambda x: int(re.sub(fr".csv", "", x))
        if re.sub(fr".csv", "", x).isdigit()
        else 0,
    )):
        if f.endswith(f".csv") and re.sub(fr".csv", "", f).isnumeric():
            df = pd.read_csv(data_dir + '/' + f)
            df = df.sort_values(by=['time'])
            if preprocess_type == "default":
               df = data_preprocessor.prepare_data(df)
            t = torch.Tensor(list(df['time']))
            e = torch.Tensor(list(df['event']))
            if max_len is not None:
                t = t[:max_len]
                e = e[:max_len]
            times.append(t)
            events.append(e)
            if unix_time:
                times[-1]/=86400
            count += 1
            if dataset_size is not None:
                if count == dataset_size:
                    break
    return times, events, data_preprocessor if preprocess_type is not None else None


def load_data_simple(
        data_dir: str,
        max_len: Optional[int]
) -> List[torch.Tensor]:
    times = []
    events = []
    print(max_len)
    for f in tqdm.tqdm(sorted(
            os.listdir(data_dir),
            key=lambda x: int(re.sub(fr".csv", "", x))
            if re.sub(fr".csv", "", x).isdigit()
            else 0,
    )):
        if f.endswith(f".csv") and re.sub(fr".csv", "", f).isnumeric():
            df = pd.read_csv(data_dir + '/' + f)
            df = df.sort_values(by=['time'])
            t = torch.Tensor(list(df['time']))
            e = torch.Tensor(list(df['event']))
            print(max_len)
            if max_len is not None:
                t = t[:max_len]
                e = e[:max_len]
            times.append(torch.Tensor(list(df['time'])))
            events.append(torch.Tensor(list(df['event'])))
    return times, events
