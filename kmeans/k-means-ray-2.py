import ray
import time
import numpy as np
import subprocess
from typing import Iterator, List, Tuple

def read_hdfs_file_subprocess(hdfs_path: str) -> Iterator[str]:
    process = subprocess.Popen(["hdfs", "dfs", "-cat", hdfs_path], stdout=subprocess.PIPE, text=True)
    for line in process.stdout:
        yield line.strip()
    process.stdout.close()
    process.wait()

@ray.remote
def process_chunk(lines: List[str]) -> np.ndarray:
    data = []
    for line in lines:
        if line.startswith("num_feature"):
            continue
        values = line.split(',')
        if len(values) >= 3:
            try:
                features = [float(values[0]), float(values[1]), float(values[2])]
                data.append(features)
            except ValueError:
                continue
    return np.array(data) if data else np.array([])

@ray.remote
def compute_closest_centers(points: np.ndarray, centers: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    assignments = []
    costs = []
    for point in points:
        distances = np.sum((centers - point) ** 2, axis=1)
        closest_center = np.argmin(distances)
        min_dist = distances[closest_center]
        assignments.append(closest_center)
        costs.append(min_dist)
    return np.array(assignments), np.array(costs)

@ray.remote
def compute_new_centers(points: np.ndarray, assignments: np.ndarray, k: int) -> np.ndarray:
    centers = []
    for i in range(k):
        mask = assignments == i
        if np.any(mask):
            center = np.mean(points[mask], axis=0)
            centers.append(center)
        else:
            centers.append(points[np.random.randint(len(points))])
    return np.array(centers)

def stream_kmeans(data_path: str, k: int = 10, max_iterations: int = 20, 
                 chunk_size: int = 1000, tolerance: float = 1e-4) -> Tuple[np.ndarray, float]:
    # First pass: compute initial centers from a sample
    print("Computing initial centers...")
    sample_data = []
    sample_size = min(10000, chunk_size * 10)  # Use a reasonable sample size
    
    lines = []
    for line in read_hdfs_file_subprocess(data_path):
        lines.append(line)
        if len(lines) >= chunk_size:
            chunk = ray.get(process_chunk.remote(lines))
            if len(chunk) > 0:
                sample_data.append(chunk)
            lines = []
            if sum(len(d) for d in sample_data) >= sample_size:
                break
    
    if lines:
        chunk = ray.get(process_chunk.remote(lines))
        if len(chunk) > 0:
            sample_data.append(chunk)
    
    initial_data = np.concatenate(sample_data)
    centers = initial_data[np.random.choice(len(initial_data), k, replace=False)]
    
    # Iterative k-means
    for iteration in range(max_iterations):
        print(f"\nIteration {iteration + 1}")
        chunk_assignments = []
        chunk_costs = []
        chunk_weighted_centers = []
        chunk_counts = []
        
        # Process data in chunks
        lines = []
        for line in read_hdfs_file_subprocess(data_path):
            lines.append(line)
            if len(lines) >= chunk_size:
                chunk = ray.get(process_chunk.remote(lines))
                if len(chunk) > 0:
                    assignments, costs = ray.get(compute_closest_centers.remote(chunk, centers))
                    chunk_assignments.append(assignments)
                    chunk_costs.append(costs)
                    
                    # Compute partial centers
                    for i in range(k):
                        mask = assignments == i
                        if np.any(mask):
                            weighted_center = np.sum(chunk[mask], axis=0)
                            count = np.sum(mask)
                            chunk_weighted_centers.append((i, weighted_center))
                            chunk_counts.append((i, count))
                
                lines = []
        
        if lines:
            chunk = ray.get(process_chunk.remote(lines))
            if len(chunk) > 0:
                assignments, costs = ray.get(compute_closest_centers.remote(chunk, centers))
                chunk_assignments.append(assignments)
                chunk_costs.append(costs)
                
                for i in range(k):
                    mask = assignments == i
                    if np.any(mask):
                        weighted_center = np.sum(chunk[mask], axis=0)
                        count = np.sum(mask)
                        chunk_weighted_centers.append((i, weighted_center))
                        chunk_counts.append((i, count))
        
        # Compute new centers
        new_centers = np.zeros_like(centers)
        cluster_counts = np.zeros(k)
        
        for i, weighted_center in chunk_weighted_centers:
            new_centers[i] += weighted_center
        for i, count in chunk_counts:
            cluster_counts[i] += count
        
        # Avoid division by zero
        mask = cluster_counts > 0
        new_centers[mask] = new_centers[mask] / cluster_counts[mask, np.newaxis]
        new_centers[~mask] = centers[~mask]  # Keep old centers for empty clusters
        
        # Check convergence
        center_shift = np.sum((centers - new_centers) ** 2)
        centers = new_centers
        
        total_cost = sum(np.sum(costs) for costs in chunk_costs)
        print(f"Cost: {total_cost:.2f}")
        
        if center_shift < tolerance:
            print("Converged!")
            break
    
    return centers, total_cost

ray.init(address="auto")

index = "1"
#data_path = "/datasets/dummy-10k.csv"
data_path = f"/datasets/dummy-data-{index}.csv"
save_path = f"/kmeans/kmeans-model-ray-{index}"
try:
    print("Starting K-means clustering...")
    start_time = time.time()
    
    centers, wssse = stream_kmeans(
        data_path=data_path,
        k=10,
        max_iterations=20,
        chunk_size=1000
    )
    
    train_time = time.time() - start_time
    print(f"\nTraining time: {train_time:.2f} seconds")
    print(f"WSSSE: {wssse:.2f}")
    
    print("\nCluster Centers:")
    for i, center in enumerate(centers):
        print(f"Cluster {i}: {center}")
    
    # Save model
    start_time = time.time()
    model_data = {"centers": centers, "wssse": wssse}
    np.save("kmeans_model.npy", model_data)
    subprocess.run(["hdfs", "dfs", "-put", "-f", "kmeans_model.npy", save_path])
    subprocess.run(["rm", "kmeans_model.npy"])
    save_time = time.time() - start_time
    print(f"Model saving time: {save_time:.2f} seconds")
    
finally:
    ray.shutdown()
