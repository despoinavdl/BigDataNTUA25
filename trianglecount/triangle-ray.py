import ray
import networkx as nx
import logging
import time
import sys
from pyarrow import fs
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up Hadoop classpath
classpath = os.popen('hadoop classpath --glob').read().strip()
os.environ["CLASSPATH"] = classpath

@ray.remote
def process_chunk(lines):
    """
    Process a chunk of lines into a NetworkX subgraph
    
    Args:
        lines: List of edge lines to process
    
    Returns:
        NetworkX graph object
    """
    G = nx.Graph()
    for line in lines:
        nodes = line.split()
        if len(nodes) == 2 and nodes[0] != "u":
            G.add_edge(nodes[0], nodes[1])
    return G

@ray.remote
def compute_triangles(G, nodes):
    """
    Compute triangles for a subset of nodes
    
    Args:
        G: NetworkX graph object
        nodes: List of nodes to process
    
    Returns:
        Dictionary of node triangle counts
    """
    return nx.triangles(G, nodes=nodes)

def process_graph(hdfs_path: str, chunk_size: int = 1000):
    """
    Process graph data and compute triangle counts
    
    Args:
        hdfs_path: Path to input file in HDFS
        chunk_size: Size of chunks for parallel processing
    
    Returns:
        Tuple of (combined triangle counts, timing measurements)
    """
    # Read data from HDFS
    load_start = time.time()
    try:
        hdfs_fs = fs.HadoopFileSystem.from_uri("hdfs://okeanos-master:54310")
        
        # Read the file content
        with hdfs_fs.open_input_stream(hdfs_path) as f:
            content = f.read().decode('utf-8')
        lines = content.splitlines()
        
    except Exception as e:
        logger.error(f"Error reading file from HDFS: {str(e)}")
        ray.shutdown()
        sys.exit(1)
    
    load_time = time.time() - load_start
    
    # Preprocessing: Create graph
    preprocessing_start = time.time()
    
    # Process file in parallel using Ray
    chunk_futures = []
    for i in range(0, len(lines), chunk_size):
        chunk = lines[i:i + chunk_size]
        chunk_futures.append(process_chunk.remote(chunk))

    # Merge results
    subgraphs = ray.get(chunk_futures)
    G = nx.compose_all(subgraphs)
    
    preprocessing_time = time.time() - preprocessing_start
    
    # Execute triangle counting
    execution_start = time.time()
    
    G_reference = ray.put(G)
    nodes = list(G.nodes)
    node_chunks = [nodes[i:i + chunk_size] for i in range(0, len(nodes), chunk_size)]
    
    # Compute triangles in parallel
    triangle_futures = [compute_triangles.remote(G_reference, nodes=node_chunk) 
                       for node_chunk in node_chunks]
    results = ray.get(triangle_futures)
    
    # Combine results
    combined_triangle_counts = {}
    for result in results:
        combined_triangle_counts.update(result)
        
    execution_time = time.time() - execution_start
    total_time = load_time + preprocessing_time + execution_time
    
    timing = {
        'load_time': load_time,
        'preprocessing_time': preprocessing_time,
        'execution_time': execution_time,
        'total_time': total_time
    }
    
    return combined_triangle_counts, timing


ray.init()

# Get dataset file name from command line
dataset_name = sys.argv[1]
hdfs_path = f"/datasets/tmp-graph-{dataset_name}.txt"

# Process graph
triangle_counts, timing = process_graph(hdfs_path)

# Write results
try:
    with open(f"/home/user/triangle/results/triangle-ray-data-{dataset_name}.output", "w") as f:
        f.write(f"Loading Time: {timing['load_time']}\n")
        f.write(f"Preprocessing Time: {timing['preprocessing_time']}\n")
        f.write(f"Execution Time: {timing['execution_time']}\n")
        f.write(f"Total Time: {timing['total_time']}\n")
        
    # Print timing information
    print(f"Loading Time: {timing['load_time']}")
    print(f"Preprocessing Time: {timing['preprocessing_time']}")
    print(f"Execution Time: {timing['execution_time']}")
    print(f"Total Time: {timing['total_time']}")
    
except Exception as e:
    logger.error(f"Error writing results: {str(e)}")

finally:
    try:
        time.sleep(2)
        ray.shutdown()
    except:
        pass
