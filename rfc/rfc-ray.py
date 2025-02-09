import ray
import sys
import time
import os 
from ray.train import ScalingConfig, FailureConfig
from ray.train.xgboost import XGBoostTrainer
from pyarrow import fs
from ray.data.preprocessors import StandardScaler
from ray.train import ScalingConfig
from sklearn.preprocessing import LabelEncoder

classpath = os.popen('hadoop classpath --glob').read().strip()
os.environ["CLASSPATH"] = classpath

if(len(sys.argv) != 3):
    print("Usage: python3 rfc-spark.py <dataset_index> <num_executors>")
    exit(1)

def transform_batch(batch):
    import pandas as pd

    batch['new_feature'] = batch['num_feature_1'] + batch['category_1'].str.len()
    return batch[
        (batch['new_feature'] > 11) & 
        (batch['num_feature_2'] + batch['num_feature_3'] < 0.2)
    ]

ray.init(runtime_env={
        "env_vars": {
            "RAY_memory_monitor_refresh_ms": "400",
            "RAY_memory_usage_threshold": "0.95"
        }})

load_start = time.time()

try:
    # Create HDFS filesystem - simpler version
    hdfs_fs = fs.HadoopFileSystem.from_uri("hdfs://okeanos-master:54310")
    
    index = sys.argv[1]
    hdfs_path = f"/datasets/dummy-data-{index}.csv"
    
    # First read the CSV
    ds = ray.data.read_csv(
        hdfs_path,
        filesystem=hdfs_fs
    )

    load_time = time.time() - load_start

    preprocessing_start = time.time()
    # Then immediately filter using map_batches to reduce memory footprint
    ds = ds.map_batches(
        transform_batch,
        batch_format="pandas",
        batch_size=1000  # Adjust this value based on your memory constraints
    )

    
    # Select columns right after filtering to further reduce memory usage
    ds = ds.repartition(500)
    ds = ds.select_columns(["num_feature_1", "num_feature_2", "num_feature_3", "new_feature", "label"])

    # Repartition before scaling to manage memory better
    # ds = ds.repartition(num_partitions=2)  # Adjust number based on dataset size
    
except Exception as e:
    logger.error(f"Error reading CSV from HDFS: {str(e)}")
    ray.shutdown()
    sys.exit(1)

# rows_before = ds.count()
# print(f"Before filtering: {rows_before}")


# rows_after = ds.count()
# print(f"After filtering: {rows_after}")

# ray.shutdown()
# exit(1)
ds = ds.repartition(500)
scaler = StandardScaler(["num_feature_1", "num_feature_2", "num_feature_3", "new_feature"])
ds = scaler.fit_transform(ds)

# print(ds.stats())
ds = ds.repartition(500)
print("Starting train-test split...")
train_ds, val_ds = ds.train_test_split(0.3)

preprocessing_time = time.time() - preprocessing_start

train_start = time.time()
trainer = XGBoostTrainer(
    scaling_config=ScalingConfig(
        num_workers=int(sys.argv[2]),
        use_gpu=False,
    ),
    label_column="label",
    num_boost_round=1,
    params = {
        "colsample_bynode": 0.8,
        "learning_rate": 1,
        "max_depth": 3,
        "num_parallel_tree": 4,
        "objective": "binary:logistic",
        "subsample": 0.8,
        "tree_method": "hist",
    },
    datasets={"train": train_ds, "valid": val_ds},
    run_config=ray.train.RunConfig(
        storage_filesystem=hdfs_fs,
        storage_path="/rfc/ray_results",
        failure_config=FailureConfig(max_failures=3),
    ),
)
result = trainer.fit()
train_time = time.time() - train_start

total_time = load_time + preprocessing_time + train_time

try:
    time.sleep(2)
    ray.shutdown()
except:
    pass

f = open(f"/home/user/rfc/results/rfc-ray-data-{index}.output", "w")
# f.write(f"Number of rows before filtering: {rows_before}\n")
# f.write(f"Number of rows after filtering: {rows_after}\n")
f.write(f"Loading Time: {load_time}\n")
f.write(f"Preproccessing Time: {preprocessing_time}\n")
f.write(f"Training Time: {train_time}\n")
f.write(f"Total Time: {total_time}\n")
f.close()

# Maybe add a small sleep to let JVM cleanup happen
time.sleep(1)


# print(f"Number of rows before filtering: {rows_before}")
# print(f"Number of rows after filtering: {rows_after}")
print(f"Loading Time: {load_time}")
print(f"Preproccessing Time: {preprocessing_time}")
print(f"Training Time: {train_time}")
print(f"Total Time: {total_time}")
