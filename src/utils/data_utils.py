from typing import List
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

def load_data_parquet(
    data_path: str,
    time_col: str = "time",
    event_col: str = "event",
    seq_col: str = "sequence_id",
) -> List[torch.Tensor]:
    """
    Reads parquet file with data of all sequences, and separates them into times/events
    """
    times = []
    events = []
    full_data = pd.read_parquet(data_path, engine="pyarrow")
    sequences = [x.reset_index(drop=True) for _, x in full_data.groupby([seq_col])]
    
    for seq in tqdm.tqdm(sequences):
            times.append(torch.Tensor(list(seq[time_col])))
            events.append(torch.Tensor(list(seq[event_col])))

    return times, events

def load_data(
    data_dir: str,
    unix_time: bool = False,
    preprocess_type: str = "default"
    ) -> List[torch.Tensor]:
    times = []
    events = []
    if preprocess_type == "default":
        data_preprocessor = Data_preprocessor()

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
            times.append(torch.Tensor(list(df['time'])))
            events.append(torch.Tensor(list(df['event'])))
            if unix_time:
                times[-1] /= 86400
    return times, events
