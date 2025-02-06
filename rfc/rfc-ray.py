import ray
from ray.train.xgboost import XGBoostTrainer, XGBoostCheckpoint
from ray.train import ScalingConfig, RunConfig, FailureConfig
import ray.data
import subprocess
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import time
from io import StringIO
from pyarrow import fs

import os
os.environ["RAY_record_ref_creation_sites"] = "1"
classpath = os.popen('hadoop classpath --glob').read().strip()
os.environ["CLASSPATH"] = classpath

# Initialize Ray cluster with YARN
ray.init()

time.sleep(5)
# Path to HDFS file
hdfs_path = "/datasets/dummy-data-1.csv"

# Function to read HDFS file line-by-line using subprocess
def read_hdfs_file_subprocess(hdfs_path):
    process = subprocess.Popen(["hdfs", "dfs", "-cat", hdfs_path], stdout=subprocess.PIPE, text=True)
    for line in process.stdout:
        yield line.strip()
    process.stdout.close()
    process.wait()

start_time = time.time()

# Read and parse the CSV header
lines = read_hdfs_file_subprocess(hdfs_path)
header = next(lines).split(",")

# Function to process lines into DataFrame chunks
@ray.remote
def process_chunk(lines_chunk):
    data_str = "\n".join(lines_chunk)
    df_chunk = pd.read_csv(StringIO(data_str), names=header)
    return df_chunk

chunk_size = 10000  # Adjust chunk size as needed
lines_buffer = []
chunk_futures = []

for line in lines:
    lines_buffer.append(line)
    if len(lines_buffer) >= chunk_size:
        chunk_futures.append(process_chunk.remote(lines_buffer))
        lines_buffer = []

# Process remaining lines
if lines_buffer:
    chunk_futures.append(process_chunk.remote(lines_buffer))

# Combine all chunks into a single DataFrame
df_parts = ray.get(chunk_futures)
df = pd.concat(df_parts, ignore_index=True)

loading_time = time.time() - start_time
print(f"Data loading took: {loading_time:.2f} seconds")

# Preprocessing
def preprocess(df):
    label_encoders = {}
    for col in ['category_1', 'category_2']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    return df, label_encoders

preprocess_start = time.time()
df, label_encoders = preprocess(df)

preprocess_time = time.time() - preprocess_start
print(f"Data preprocessing took: {preprocess_time:.2f} seconds")

# Convert DataFrame to Ray Dataset
ray_dataset = ray.data.from_pandas(df)

# Training configuration
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'tree_method': 'hist',  # Optimized for large datasets
    'num_parallel_tree': 8  # Emulating Random Forest with multiple trees
}

hdfs_fs = fs.HadoopFileSystem.from_uri("hdfs://okeanos-master:54310")

trainer = XGBoostTrainer(
    params=params,
    datasets={"train": ray_dataset},
    label_column="label",
    num_boost_round=1,
    scaling_config=ScalingConfig(
        num_workers=2,
        resources_per_worker={"CPU": 3},  # Specify CPU per worker
        placement_strategy="SPREAD"  # This encourages spreading across nodes
    ),
    run_config=RunConfig(
        failure_config=FailureConfig(max_failures=0), 
        storage_filesystem=hdfs_fs,
        storage_path="/rfc/ray_results"
    )
)

train_start = time.time()
result = trainer.fit()
train_time = time.time() - train_start
print(f"Model training took: {train_time:.2f} seconds")

print("Training completed.")
