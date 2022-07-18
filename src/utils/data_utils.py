from typing import List
import os
import tqdm
import torch
import pandas as pd
import re

def load_data(
    data_dir: str,
    unix_time = False
    ) -> List[torch.Tensor]:
    times = []
    events = []
    for f in tqdm.tqdm(sorted(
        os.listdir(data_dir),
        key=lambda x: int(re.sub(fr".csv", "", x))
        if re.sub(fr".csv", "", x).isdigit()
        else 0,
    )):
        if f.endswith(f".csv") and re.sub(fr".csv", "", f).isnumeric():
            df = pd.read_csv(data_dir + '/' + f)
            df = df.sort_values(by=['time'])
            times.append(torch.Tensor(list(df['time'])))
            events.append(torch.Tensor(list(df['event'])))
            if unix_time:
                times[-1]/=86400
    return times, events
