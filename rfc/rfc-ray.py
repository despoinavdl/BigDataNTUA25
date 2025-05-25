# CLASSPATH=`$HADOOP_HOME/bin/hdfs classpath --glob` ray start --head --port=6379 --num-cpus=4 --object-store-memory=2000000000
# CLASSPATH=`$HADOOP_HOME/bin/hdfs classpath --glob` ray start --address='192.168.0.1:6379' --num-cpus=4 --object-store-memory=2000000000
import ray
import sys
import time
import os 
import logging
from ray.train import ScalingConfig, FailureConfig
from ray.train.xgboost import XGBoostTrainer, RayTrainReportCallback
from pyarrow import fs
from ray.data.preprocessors import StandardScaler
from ray.train import ScalingConfig

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

classpath = os.popen('hadoop classpath --glob').read().strip()
os.environ["CLASSPATH"] = classpath

log_messages = []

def log_to_buffer(message):
    """Add a message to the log buffer and also log it through the logger"""
    log_messages.append(message)
    logger.info(message)

if(len(sys.argv) != 3):
    print("Usage: python3 rfc-ray-test-mine.py <dataset_index> <num_chunks>")
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
    hdfs_fs = fs.HadoopFileSystem.from_uri("hdfs://okeanos-master:54310")
    
    index = sys.argv[1]
    num_chunks = int(sys.argv[2])
    hdfs_path = f"/datasets/dummy-data-{index}.csv"
    
    logger.info(f"Running with index={index}, num_chunks={num_chunks}")
    
    ds = ray.data.read_csv(
        hdfs_path,
        filesystem=hdfs_fs
    )

    load_time = time.time() - load_start
    log_to_buffer(f"Dataset name: dummy-data-{index}.csv, Number of chunks: {num_chunks}\n")
    log_to_buffer(f"Loading Time: {load_time:.2f} seconds")

    preprocessing_start = time.time()
    
    # Calculate dataset size and get batch size from num_chunks
    dataset_size = ds.count()
    batch_size = max(1, dataset_size // num_chunks)
    logger.info(f"Dataset size: {dataset_size}, Using {num_chunks} chunks with batch size: {batch_size}")
    
    # Filter using map_batches with calculated batch_size
    ds = ds.map_batches(
        transform_batch,
        batch_format="pandas",
        batch_size=batch_size,
        concurrency=2
    )

    ds = ds.repartition(16)
    ds = ds.select_columns(["num_feature_1", "num_feature_2", "num_feature_3", "new_feature", "label"])

except Exception as e:
    logger.error(f"Error reading CSV from HDFS: {str(e)}")
    ray.shutdown()
    sys.exit(1)

ds = ds.repartition(16)
scaler = StandardScaler(["num_feature_1", "num_feature_2", "num_feature_3", "new_feature"])
ds = scaler.fit_transform(ds)

# print(ds.stats())
ds = ds.repartition(16)
print("Starting train-test split...")
train_ds, val_ds = ds.train_test_split(0.3)

preprocessing_time = time.time() - preprocessing_start
log_to_buffer(f"Dataset pre-processing time: {preprocessing_time:.2f} seconds")

train_start = time.time()
trainer = XGBoostTrainer(
    scaling_config=ScalingConfig(
        num_workers=2,
        use_gpu=False,
        resources_per_worker={"CPU": 3}
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
log_to_buffer(f"Train Time: {train_time:.2f} seconds")

op_start = time.time()
booster = RayTrainReportCallback.get_model(result.checkpoint)
val_df = val_ds.to_pandas()
feature_columns = ["num_feature_1", "num_feature_2", "num_feature_3", "new_feature"]
X_val = val_df[feature_columns]
import xgboost as xgb
dmatrix = xgb.DMatrix(X_val)
predictions = booster.predict(dmatrix)
log_to_buffer(f"Made {len(predictions)} predictions")
op_time = time.time() - op_start
log_to_buffer(f"Operation Time: {op_time:.2f} seconds")


total_time = load_time + preprocessing_time + train_time + op_time
log_to_buffer(f"Total Time: {total_time:.2f} seconds\n\n")

try:
    timestamp = time.strftime("%H%M%S")
    log_path = f"/user/user/rfc-ray/output-log-dummy-data-{index}-{timestamp}"
    
    log_content = "\n".join(log_messages)
    
    with hdfs_fs.open_output_stream(log_path) as f:
        f.write(log_content.encode('utf-8'))
    
    logger.info(f"Log saved to HDFS: {log_path}")
except Exception as e:
    logger.error(f"Error saving logs to HDFS: {str(e)}")

try:
    time.sleep(2)
    ray.shutdown()
except:
    pass

time.sleep(1)
