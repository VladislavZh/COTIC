from typing import List
import os
import tqdm
import torch
import pandas as pd

def load_data(
    path_to_data: str,
    n_types: int,
    unix_time = False
    ) -> List[torch.Tensor]:
    files = os.listdir(path_to_data)
    times = []
    events = []
    if 'clusters.csv' in files:
        files.remove('clusters.csv')
    if 'process_params.json' in files:
        files.remove('process_params.json')
    for i, f in tqdm.tqdm(enumerate(files)):
        df = pd.read_csv(path_to_data + '/' + f)
        df = df.sort_values(by=['time'])
        times.append(torch.Tensor(list(df['time'])))
        events.append(torch.Tensor(list(df['event'])))
        if unix_time:
            times[-1]/=86400
    return times, events
