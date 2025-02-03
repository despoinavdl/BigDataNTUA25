import ray
import networkx as nx
import subprocess
import time

# Path to HDFS file
hdfs_path = "/datasets/tmp-graph.txt"

# Function to read HDFS file line-by-line using subprocess
def read_hdfs_file_subprocess(hdfs_path):
    process = subprocess.Popen(["hdfs", "dfs", "-cat", hdfs_path], stdout=subprocess.PIPE, text=True)
    for line in process.stdout:
        yield line.strip()
    process.stdout.close()
    process.wait()

ray.init()

start_time = time.time()

# Ray remote function to process lines into a NetworkX subgraph
@ray.remote
def process_chunk(lines):
    G = nx.Graph()
    for line in lines:
        nodes = line.split()  # Adjust separator if needed
        if len(nodes) == 2 and nodes[0] != "u":   # Ensure valid edge format
            G.add_edge(nodes[0], nodes[1])
    return G

# Process file in parallel using Ray
chunk_size = 1000
lines = []
chunk_futures = []

for line in read_hdfs_file_subprocess(hdfs_path):
    lines.append(line)
    if len(lines) >= chunk_size:
        chunk_futures.append(process_chunk.remote(lines))
        lines = []

# Process remaining lines
if lines:
    chunk_futures.append(process_chunk.remote(lines))

# Merge results
subgraphs = ray.get(chunk_futures)
G = nx.compose_all(subgraphs)

print(G)
creation_time = time.time() - start_time
print(f"Graph creation took: {creation_time:.2f} seconds")

@ray.remote
def compute_triangles(G, nodes):
    return nx.triangles(G, nodes=nodes)

start_exec = time.time()
G_reference = ray.put(G)

# Split the nodes into chunks for parallel processing
nodes = list(G.nodes)
node_chunks = [nodes[i:i + chunk_size] for i in range(0, len(nodes), chunk_size)]
num_chunks = len(node_chunks)

# Compute triangles in parallel for each chunk
results = [ray.get(compute_triangles.remote(G_reference, nodes=node_chunk)) for node_chunk in node_chunks]

exec_time = time.time() - start_exec
print(f"Execution time: {exec_time:.2f} seconds")

combined_triangle_counts = {}
for result in results:
    combined_triangle_counts.update(result)

# Pretty print the combined triangle counts
print("\nNode Triangle Counts:")
print("-" * 40)
print(f"{'Node':<20}{'Triangle Count':<20}")
print("-" * 40)

for node, count in combined_triangle_counts.items():
    print(f"{node:<20}{count:<20}")
    
print("-" * 40)
