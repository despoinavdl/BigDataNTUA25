# python3 test.py <index> <num_chunks>
import ray
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any
from ray.data.preprocessors import StandardScaler
from ray.data import Dataset, DataContext
from pyarrow import fs
import logging
import time
import sys
import os
import gc

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure Hadoop classpath
try:
    classpath = os.popen('hadoop classpath --glob').read().strip()
    os.environ["CLASSPATH"] = classpath
    logger.info("Hadoop classpath configured successfully")
except Exception as e:
    logger.error(f"Failed to configure Hadoop classpath: {str(e)}")

# Log buffer for collecting performance metrics
log_messages = []

def log_to_buffer(message):
    """Add a message to the log buffer and also log it through the logger"""
    log_messages.append(message)
    logger.info(message)

# Disable tensor casting to improve memory usage
ctx = DataContext.get_current()
ctx.enable_tensor_extension_casting = False

#-------------------- K-means Core Functions --------------------#

def find_nearest_center(batch: Dict[str, np.ndarray], centers: List[np.ndarray]):
    """
    Assign each point in the batch to nearest center using vectorized operations.
    
    Args:
        batch: Dictionary containing feature arrays
        centers: List of center coordinates
    
    Returns:
        Updated batch with cluster assignments
    """
    try:
        features = np.column_stack([batch[f] for f in batch.keys() if f != 'cluster'])
        distances = np.array([np.sum((features - c) ** 2, axis=1) for c in centers])
        batch['cluster'] = np.argmin(distances, axis=0)
        return batch
    except Exception as e:
        logger.error(f"Error in find_nearest_center: {str(e)}")
        raise

def update_centers(dataset: Dataset, n_clusters: int, n_features: int):
    """
    Calculate new center positions using groupby aggregation
    
    Args:
        dataset: Dataset with cluster assignments
        n_clusters: Number of clusters
        n_features: Number of features
        
    Returns:
        List of new center coordinates
    """
    try:
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
                logger.warning(f"Empty cluster {i} detected. Using zero vector as center.")
                centers.append(np.zeros(n_features))
        
        # Force garbage collection to free memory
        del result_df
        gc.collect()
        
        return centers
    except Exception as e:
        logger.error(f"Error in update_centers: {str(e)}")
        raise

def kmeans_ray(scaled_ds: Dataset, feature_columns: List[str], n_clusters: int = 2, 
               max_iters: int = 10, num_chunks: int = 10, convergence_threshold: float = 1e-5):
    """
    Perform K-means clustering using Ray.
    
    Args:
        scaled_ds: Input Ray dataset
        feature_columns: List of feature column names
        n_clusters: Number of clusters
        max_iters: Maximum number of iterations
        num_chunks: Number of chunks to divide the dataset into
        convergence_threshold: Threshold for convergence check
    
    Returns:
        Tuple of (clustered dataset, final centers)
    """
    try:
        # Initialize centers randomly
        initial_points = scaled_ds.take(n_clusters)
        centers = [np.array([point[col] for col in feature_columns]) 
                   for point in initial_points]
        
        logger.info(f"Initial centers: {centers}")
        
        # Calculate total dataset size and derive batch size from num_chunks
        dataset_size = scaled_ds.count()
        batch_size = max(1, dataset_size // num_chunks)
        logger.info(f"Dataset size: {dataset_size}, Using {num_chunks} chunks with batch size: {batch_size}")
        
        # Main clustering loop
        for iteration in range(max_iters):
            iteration_start = time.time()
            
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
            
            iteration_time = time.time() - iteration_start
            logger.info(f"Iteration {iteration + 1} completed in {iteration_time:.2f} seconds")
            
            # Check convergence
            if np.allclose(centers, new_centers, rtol=convergence_threshold):
                logger.info(f"Converged after {iteration + 1} iterations")
                break
                
            centers = new_centers
            
            # Force garbage collection after each iteration
            gc.collect()
        
        return clustered_ds, centers
    except Exception as e:
        logger.error(f"Error in kmeans_ray: {str(e)}")
        raise

#-------------------- Main Execution --------------------#

def main():
    try:
        # Initialize Ray
        ray.init(ignore_reinit_error=True)
        logger.info("Ray initialized successfully")
        
        # Read command line arguments
        if len(sys.argv) < 3:
            logger.error("Missing required arguments. Usage: python script.py <index> <num_chunks>")
            sys.exit(1)
            
        index = sys.argv[1]
        num_chunks = int(sys.argv[2])
        
        logger.info(f"Running with index={index}, num_chunks={num_chunks}")
        feature_cols = ['num_feature_1', 'num_feature_2', 'num_feature_3']
        
        # Read data from HDFS
        load_start = time.time()
        try:
            # Create HDFS filesystem
            hdfs_fs = fs.HadoopFileSystem.from_uri("hdfs://okeanos-master:54310")
            hdfs_path = f"/datasets/dummy-data-{index}.csv"
            
            logger.info(f"Reading data from HDFS: {hdfs_path}")
            
            # Read CSV using Ray with the filesystem
            ds = ray.data.read_csv(
                hdfs_path,
                filesystem=hdfs_fs
            )
            
        except Exception as e:
            logger.error(f"Error reading CSV from HDFS: {str(e)}")
            raise
            
        load_time = time.time() - load_start
        log_to_buffer(f"Loading Time: {load_time:.2f} seconds")
        
        # Preprocess data
        preprocessing_start = time.time()
        try:
            ds = ds.select_columns(feature_cols)
            # Normalize features
            scaler = StandardScaler(columns=feature_cols)
            scaled_ds = scaler.fit_transform(ds)
            # Free memory
            del ds
            gc.collect()
        except Exception as e:
            logger.error(f"Error during preprocessing: {str(e)}")
            raise
            
        preprocessing_time = time.time() - preprocessing_start
        log_to_buffer(f"Dataset pre-processing time: {preprocessing_time:.2f} seconds")
        
        # Run clustering
        train_start = time.time()
        try:
            result_ds, final_centers = kmeans_ray(
                scaled_ds=scaled_ds,
                feature_columns=feature_cols,
                n_clusters=2,
                max_iters=3,
                num_chunks=num_chunks
            )
        except Exception as e:
            logger.error(f"Error during clustering: {str(e)}")
            raise
            
        train_time = time.time() - train_start
        log_to_buffer(f"Execution Time: {train_time:.2f} seconds")
        
        # Show results
        op_start = time.time()
        try:
            logger.info(f"Final centers: {final_centers}")
            result_ds.show()
        except Exception as e:
            logger.error(f"Error showing results: {str(e)}")
            
        op_time = time.time() - op_start
        log_to_buffer(f"Operation Time: {op_time:.2f} seconds")
        
        # Calculate total time
        total_time = load_time + preprocessing_time + train_time + op_time
        log_to_buffer(f"Total Time: {total_time:.2f} seconds\n\n")
        log_to_buffer(f"Dataset name: dummy-data-{index}.csv, Number of chunks: {num_chunks}\n")
        
        # Save logs to HDFS
        try:
            timestamp = time.strftime("%H%M%S")
            log_path = f"/user/user/kmeans-ray/output-log-dummy-data-{index}-{timestamp}"
            
            # Convert log messages to a single string
            log_content = "\n".join(log_messages)
            
            # Write to HDFS
            with hdfs_fs.open_output_stream(log_path) as f:
                f.write(log_content.encode('utf-8'))
            
            logger.info(f"Log saved to HDFS: {log_path}")
        except Exception as e:
            logger.error(f"Error saving logs to HDFS: {str(e)}")
            
    except Exception as e:
        logger.error(f"Unhandled exception in main: {str(e)}")
    finally:
        # Clean up
        try:
            ray.shutdown()
            logger.info("Ray shutdown completed")
        except Exception as e:
            logger.error(f"Error during Ray shutdown: {str(e)}")
        
        # Final garbage collection
        gc.collect()

if __name__ == "__main__":
    main()
