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

    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame: #number_max, number_min
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
            #self.min_max.fit(np.array([number_max, number_min]).reshape(-1, 1))
            data['time'] = np.squeeze(self.min_max.fit_transform(data['time'].values.reshape(-1, 1)))

        return data

def load_data(
    data_dir: str,
    unix_time: bool = False,
    preprocess_type: str = "default"
    ) -> List[torch.Tensor]:
    times = []
    events = []
    number_max = 0
    number_min = 0
    number_max_dt = 0
    number_min_dt = 0
    list_of_time = []
    if preprocess_type == "default":
        data_preprocessor = Data_preprocessor()

        for f in sorted(
            os.listdir(data_dir),
            key=lambda x: int(re.sub(fr".csv", "", x))
            if re.sub(fr".csv", "", x).isdigit()
            else 0,
        ):
            if f.endswith(f".csv") and re.sub(fr".csv", "", f).isnumeric():
                df = pd.read_csv(data_dir + '/' + f)
                df = df.sort_values(by=['time'])
                if max(df['time']) > number_max:
                    number_max = max(df['time'])
                if min(df['time']) > number_min:
                    number_min = min(df['time'])

                all_differencies = []
                interm_df = df.groupby('event').diff()
                
                if max(np.nan_to_num(interm_df['time'])) > number_max_dt:
                    number_max_dt = max(np.nan_to_num(interm_df['time']))
                if min(np.nan_to_num(interm_df['time'])) > number_min_dt:
                    number_min_dt = min(np.nan_to_num(interm_df['time']))
                for i in range(len(df['time'])):
                    list_of_time.append(df['time'][i])

    #print(len(list_of_time))
    #number_quantile_95 = np.quantile(list_of_time, 0.95)
    #number_quantile_05 = np.quantile(list_of_time, 0.05)
    #print(number_quantile_95, number_quantile_05)

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
                times[-1]/=86400
    return times, events, data_preprocessor.min_max if preprocess_type == "default" else None
