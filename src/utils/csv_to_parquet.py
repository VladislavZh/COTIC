import argparse
import glob
import os
import re

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset-name", default="ATM")
args = vars(parser.parse_args())
data_folder = "data"
dataset = args["dataset_name"]
time_col = "time"
event_col = "event"

csvs = glob.glob(os.path.join(data_folder, dataset, "*.csv"))
# filter list by files containing digits
csvs = [c for c in csvs if bool(re.search("\d", c))]

seq_id = csvs[0].split("/")[-1]
seq_id = seq_id.split(".")[0]
total_df = pd.read_csv(csvs[0], usecols=[time_col,event_col])
total_df["sequence_id"] = seq_id

for i in tqdm.tqdm(range(1, len(csvs)), total=len(csvs), miniters=1):
    seq_id = csvs[i].split("/")[-1]
    seq_id = seq_id.split(".")[0]
    curr_df = pd.read_csv(csvs[i], usecols=[time_col,event_col])
    curr_df["sequence_id"] = seq_id
    total_df = pd.concat([total_df, curr_df])

print(total_df.event.unique())
# compressing datatypes
total_df[time_col] = total_df[time_col].astype("float32")
if total_df.dtypes[event_col] == "int64":
    total_df[event_col] = total_df[event_col].astype("int16")
total_df.reset_index(inplace=True, drop=True)
print(total_df.info())
print(total_df.sample(n=5))

# saving to parquet
pq_table = pa.Table.from_pandas(total_df)
pq.write_table(pq_table, os.path.join(data_folder, dataset + ".parquet"))
