from typing import List
import os
import tqdm
import torch
import pandas as pd

def load_data(
    path_to_data: str,
    unix_time = False
    ) -> List[torch.Tensor]:
    times = []
    events = []
    for file in tqdm.tqdm(sorted(
        os.listdir(data_dir),
        key=lambda x: int(re.sub(fr".csv", "", x))
        if re.sub(fr".csv", "", x).isdigit()
        else 0,
    )):
        if file.endswith(f".csv") and re.sub(fr".csv", "", file).isnumeric():
            df = pd.read_csv(path_to_data + '/' + f)
            df = df.sort_values(by=['time'])
            times.append(torch.Tensor(list(df['time'])))
            events.append(torch.Tensor(list(df['event'])))
            if unix_time:
                times[-1]/=86400
    return times, events
