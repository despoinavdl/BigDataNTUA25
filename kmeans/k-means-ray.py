import ray
import time
import numpy as np
import subprocess

def read_hdfs_file_subprocess(hdfs_path):
    process = subprocess.Popen(["hdfs", "dfs", "-cat", hdfs_path], stdout=subprocess.PIPE, text=True)
    for line in process.stdout:
        yield line.strip()
    process.stdout.close()
    process.wait()

@ray.remote
def process_chunk(lines):
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
    return np.array(data)

@ray.remote
def compute_closest_centers(points, centers):
    assignments = []
    costs = []
    for point in points:
        min_dist = float('inf')
        closest_center = 0
        for i, center in enumerate(centers):
            dist = np.sum((point - center) ** 2)
            if dist < min_dist:
                min_dist = dist
                closest_center = i
        assignments.append(closest_center)
        costs.append(min_dist)
    return np.array(assignments), np.array(costs)

@ray.remote
def compute_new_centers(points, assignments, k):
    centers = []
    for i in range(k):
        mask = assignments == i
        if np.any(mask):
            center = np.mean(points[mask], axis=0)
            centers.append(center)
        else:
            centers.append(points[np.random.randint(len(points))])
    return np.array(centers)

def distributed_kmeans(data, k=10, max_iterations=20, tolerance=1e-4):
    n_points = len(data)
    centers = data[np.random.choice(n_points, k, replace=False)]
    
    n_chunks = int(ray.available_resources()['CPU'])
    chunk_size = len(data) // n_chunks
    chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
    
    for iteration in range(max_iterations):
        futures = [compute_closest_centers.remote(chunk, centers) for chunk in chunks]
        try:
            results = ray.get(futures)
        except Exception as e:
            print(f"Error during computation: {e}")
            ray.shutdown()
            raise e
            
        assignments = np.concatenate([r[0] for r in results])
        costs = np.concatenate([r[1] for r in results])
        total_cost = np.sum(costs)
        
        new_centers = ray.get(compute_new_centers.remote(data, assignments, k))
        center_shift = np.sum((centers - new_centers) ** 2)
        centers = new_centers
        
        print(f"Iteration {iteration + 1}, Cost: {total_cost:.2f}")
        
        if center_shift < tolerance:
            break
            
    return centers, assignments, total_cost

ray.init(address="auto")

index = "1"
data_path = f"/datasets/dummy-data-{index}.csv"
save_path = f"/kmeans/kmeans-model-ray-{index}"

start_time = time.time()
chunk_size = 1000
lines = []
chunk_futures = []

for line in read_hdfs_file_subprocess(data_path):
    lines.append(line)
    if len(lines) >= chunk_size:
        chunk_futures.append(process_chunk.remote(lines))
        lines = []

if lines:
    chunk_futures.append(process_chunk.remote(lines))

processed_chunks = ray.get(chunk_futures)
feature_array = np.concatenate(processed_chunks)
load_time = time.time() - start_time
print(f"Data loading time: {load_time:.2f} seconds")

start_time = time.time()
centers, assignments, wssse = distributed_kmeans(feature_array, k=10, max_iterations=20)
train_time = time.time() - start_time
print(f"Training time: {train_time:.2f} seconds")
print(f"WSSSE: {wssse:.2f}")

print("\nCluster Centers:")
for i, center in enumerate(centers):
    print(f"Cluster {i}: {center}")

unique, counts = np.unique(assignments, return_counts=True)
print("\nCluster Sizes:")
for cluster, count in zip(unique, counts):
    print(f"Cluster {cluster}: {count}")

start_time = time.time()
model_data = {"centers": centers, "wssse": wssse}
np.save("kmeans_model.npy", model_data)
subprocess.run(["hdfs", "dfs", "-put", "-f", "kmeans_model.npy", save_path])
subprocess.run(["rm", "kmeans_model.npy"])
save_time = time.time() - start_time
print(f"Model saving time: {save_time:.2f} seconds")

ray.shutdown()
