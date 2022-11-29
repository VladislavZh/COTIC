import glob
import os
import re

import pandas as pd
import pyarrow as pa
import tqdm

data_folder = "data"
dataset = "ATM"
time_col = "time"
event_col = "event"

csvs = glob.glob(os.path.join(data_folder, dataset, "*.csv"))
# filter list by files containing digits
csvs = [c for c in csvs if bool(re.search("\d", c))]

user_id = csvs[0].split("/")[-1]
user_id = user_id.split(".")[0]
total_df = pd.read_csv(csvs[0])
total_df = total_df[[time_col, event_col]]
total_df["user_id"] = user_id

for i in tqdm.tqdm(range(1, len(csvs)), total=len(csvs) - 1, miniters=1):
    user_id = csvs[i].split("/")[-1]
    user_id = user_id.split(".")[0]
    curr_df = pd.read_csv(csvs[i])
    curr_df = total_df[[time_col, event_col]]
    curr_df["user_id"] = user_id
    total_df = pd.concat([total_df, curr_df])


# compressing datatypes
# total_df[time_col] = total_df[time_col].astype("float16")
# if total_df.dtypes[event_col] == "int64":
#     total_df[event_col] = total_df[event_col].astype("int8")
total_df.reset_index(inplace=True)
print(total_df.info())

# saving to parquet
pq_table = pa.Table.from_pandas(total_df)
pa.parquet(pq_table, os.path.join(data_folder, dataset + ".parquet"))
