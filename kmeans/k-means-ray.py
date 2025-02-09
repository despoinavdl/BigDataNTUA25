import ray
import numpy as np
import pandas as pd  # Add this at the top with other imports
from typing import List, Dict, Tuple, Any
from ray.data.preprocessors import StandardScaler
from ray.data import Dataset
import logging
import time
import sys
from pyarrow import fs
import pyarrow.csv as csv  # Add this import
import pyarrow.dataset as ds  # And this one
import os

classpath = os.popen('hadoop classpath --glob').read().strip()
os.environ["CLASSPATH"] = classpath

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def find_nearest_center(batch: Dict[str, np.ndarray], centers: List[np.ndarray]):
    """
    Assign each point in the batch to nearest center using vectorized operations.
    
    Args:
        batch: Dictionary containing feature arrays
        centers: List of center coordinates
    
    Returns:
        Updated batch with cluster assignments
    """
    features = np.column_stack([batch[f] for f in batch.keys() if f != 'cluster'])
    distances = np.array([np.sum((features - c) ** 2, axis=1) for c in centers])
    batch['cluster'] = np.argmin(distances, axis=0)
    return batch

from ray.data import DataContext
# Disable tensor casting
ctx = DataContext.get_current()
ctx.enable_tensor_extension_casting = False

def update_centers(dataset: Dataset, n_clusters: int, n_features: int):
    """
    Calculate new center positions using groupby aggregation
    """
    def sum_points(df):
        sums = []
        counts = []
        clusters = []
        
        for i in range(n_clusters):
            mask = df['cluster'] == i
            if np.any(mask):
                features = df[mask][[col for col in df.columns if col != 'cluster']]
                sums.append(features.sum().values)
                counts.append(mask.sum())
                clusters.append(i)
                
        return pd.DataFrame({
            'cluster': clusters,
            'sum': sums,
            'count': counts
        })

    # Aggregate results
    result_df = dataset.map_batches(
        sum_points,
        batch_format="pandas"
    ).to_pandas()
    
    # Calculate centers
    centers = []
    for i in range(n_clusters):
        cluster_data = result_df[result_df['cluster'] == i]
        if not cluster_data.empty:
            total_sum = np.sum(np.stack(cluster_data['sum'].values), axis=0)
            total_count = cluster_data['count'].sum()
            centers.append(total_sum / total_count)
        else:
            centers.append(np.zeros(n_features))
            
    return centers

def kmeans_ray(scaled_ds: Dataset, feature_columns: List[str], n_clusters: int = 2, max_iters: int = 10, batch_size: int = 10000):
    """
    Perform K-means clustering using Ray.
    
    Args:
        scaled_ds: Input Ray dataset
        feature_columns: List of feature column names
        n_clusters: Number of clusters
        max_iters: Maximum number of iterations
        batch_size: Batch size for processing
    
    Returns:
        Tuple of (clustered dataset, final centers)
    """
    
    # Initialize centers randomly
    initial_points = scaled_ds.take(n_clusters)
    centers = [np.array([point[col] for col in feature_columns]) 
                for point in initial_points]
    
    logger.info(f"Initial centers: {centers}")
    
    # Main clustering loop
    for iteration in range(max_iters):
        # Assign points to clusters
        clustered_ds = scaled_ds.map_batches(
            lambda batch: find_nearest_center(batch, centers),
            batch_format="pandas",
            batch_size=batch_size
        )
        
        # Update centers
        new_centers = update_centers(
            clustered_ds,
            n_clusters,
            len(feature_columns)
        )
        
        # Check convergence
        if np.allclose(centers, new_centers, rtol=1e-5):
            logger.info(f"Converged after {iteration + 1} iterations")
            break
            
        centers = new_centers
        logger.info(f"Iteration {iteration + 1} completed")
    
    return clustered_ds, centers


ray.init()

# Read data from HDFS
load_start = time.time()
feature_cols = ['num_feature_1', 'num_feature_2', 'num_feature_3']
index = sys.argv[1]
try:
    # Create HDFS filesystem - simpler version
    hdfs_fs = fs.HadoopFileSystem.from_uri("hdfs://okeanos-master:54310")
    
    index = sys.argv[1]
    hdfs_path = f"/datasets/dummy-data-{index}.csv"
    
    # Read CSV using Ray with the filesystem, without block_size
    ds = ray.data.read_csv(
        hdfs_path,
        filesystem=hdfs_fs
    )
    
except Exception as e:
    logger.error(f"Error reading CSV from HDFS: {str(e)}")
    ray.shutdown()
    sys.exit(1)

load_time = time.time() - load_start
preprocessing_start = time.time()
ds = ds.select_columns(["num_feature_1", "num_feature_2", "num_feature_3"])
# Normalize features
scaler = StandardScaler(columns=feature_cols)
scaled_ds = scaler.fit_transform(ds)
preprocessing_time = time.time() - preprocessing_start

train_start = time.time()
# Run clustering
result_ds, final_centers = kmeans_ray(
    scaled_ds=scaled_ds,
    feature_columns=feature_cols,
    n_clusters=2,
    max_iters=3,
    batch_size=1000
)
train_time = time.time() - train_start

# Show results
logger.info(f"Final centers: {final_centers}")
result_ds.show()


total_time = load_time + preprocessing_time + train_time

try:
    time.sleep(2)
    ray.shutdown()
except:
    pass

f = open(f"/home/user/kmeans/results/kmeans-ray-data-{index}.output", "w")
f.write(f"Loading Time: {load_time}\n")
f.write(f"Preproccessing Time: {preprocessing_time}\n")
f.write(f"Training Time: {train_time}\n")
f.write(f"Total Time: {total_time}\n")
f.close()

time.sleep(1)


print(f"Loading Time: {load_time}")
print(f"Preproccessing Time: {preprocessing_time}")
print(f"Training Time: {train_time}")
print(f"Total Time: {total_time}")

