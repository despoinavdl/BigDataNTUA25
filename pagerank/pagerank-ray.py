import ray
import networkx as nx
import subprocess
import time

# Path to HDFS file
hdfs_path = "/datasets/sx-stackoverflow.txt"

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

# Compute full PageRank once, since networkx's pagerank is inherently single-threaded. 
start_pagerank_time = time.time()
full_pagerank = nx.pagerank(G)

pagerank_time = time.time() - start_pagerank_time
print(f"PageRank computation took: {pagerank_time:.2f} seconds")

# Print top 10 nodes by PageRank score
top_nodes = sorted(full_pagerank.items(), key=lambda x: x[1], reverse=True)[:10]
print("\nTop 10 Nodes by PageRank:")
for node, score in top_nodes:
    print(f"Node: {node}, PageRank: {score:.6f}")
