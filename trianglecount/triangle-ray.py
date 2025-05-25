# python3 triangle-ray-test.py <dataset_name> <num_chunks> <batch_size>
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

log_messages = []

def log_to_buffer(message):
    log_messages.append(message)
    logger.info(message)

if len(sys.argv) < 3:
    logger.error("Usage: python3 triangle-ray-test.py <dataset_name> <num_chunks> <batch_size>")
    sys.exit(1)
        

classpath = os.popen('hadoop classpath --glob').read().strip()
os.environ["CLASSPATH"] = classpath

@ray.remote
def compute_triangles(G, nodes):
    return nx.triangles(G, nodes=nodes)

def process_graph(hdfs_path: str, num_chunks: int = 10):
    # Read data from HDFS
    load_start = time.time()
    try:
        hdfs_fs = fs.HadoopFileSystem.from_uri("hdfs://okeanos-master:54310")
        
        # Read the file content
        with hdfs_fs.open_input_stream(hdfs_path) as f:
            content = f.read().decode('utf-8')
        lines = content.splitlines()[1:]
        
    except Exception as e:
        logger.error(f"Error reading file from HDFS: {str(e)}")
        ray.shutdown()
        sys.exit(1)
    
    load_time = time.time() - load_start
    
    # Preprocessing: Create graph
    preprocessing_start = time.time()
    
    G = nx.parse_edgelist(lines, nodetype=int, create_using=nx.DiGraph())
    G = G.to_undirected()

    preprocessing_time = time.time() - preprocessing_start

    logger.info(f"Graph is ready and read")
    
    # Execute triangle counting
    execution_start = time.time()
    
    G_reference = ray.put(G)
    nodes = list(G.nodes)

    # Calculate node chunk size based on number of chunks
    k, m = divmod(len(nodes), num_chunks)
    node_chunks = [nodes[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(num_chunks)]

    batch_size = len(node_chunks) if sys.argv[3] == "max" else int(sys.argv[3])  # Process this many chunks concurrently
    combined_triangle_counts = {}
    
    for i in range(0, len(node_chunks), batch_size):
        # Get the current batch of chunks
        current_batch = node_chunks[i:i + batch_size]
        
        # Create and execute futures for this batch only
        batch_futures = [compute_triangles.remote(G_reference, nodes=node_chunk) 
                       for node_chunk in current_batch]
        
        # Wait for this batch to complete before starting the next one
        batch_results = ray.get(batch_futures)
        
        # Combine results
        for result in batch_results:
            combined_triangle_counts.update(result)

        # Log progress
        processed = min((i + batch_size), len(node_chunks))
        logger.info(f"Processed {processed}/{len(node_chunks)} node chunks")

    execution_time = time.time() - execution_start

    op_start = time.time()
    total_triangles = sum(combined_triangle_counts.values()) // 3  # Each triangle is counted 3 times
    print(f"Total triangles found: {total_triangles}")
    op_time = time.time() - op_start

    timing = {
        'load_time': load_time,
        'preprocessing_time': preprocessing_time,
        'execution_time': execution_time,
        'op_time': op_time,
        'total_time': load_time + preprocessing_time + execution_time + op_time
    }
    return combined_triangle_counts, timing


ray.init()

dataset_name = sys.argv[1]
hdfs_path = f"/datasets/{dataset_name}"
num_chunks = int(sys.argv[2])

triangle_counts, timing = process_graph(hdfs_path, num_chunks)

log_to_buffer(f"Dataset name: {dataset_name}, Number of chunks: {num_chunks}, Batch size: {sys.argv[3]}")
log_to_buffer(f"Loading Time: {timing['load_time']:.2f} seconds")
log_to_buffer(f"Dataset pre-processing time: {timing['preprocessing_time']:.2f} seconds")
log_to_buffer(f"Execution Time: {timing['execution_time']:.2f} seconds")
log_to_buffer(f"Operation Time: {timing['op_time']:.2f} seconds")
log_to_buffer(f"Total Time: {timing['total_time']:.2f} seconds\n\n")

timestamp = time.strftime("%H%M%S")
log_path = f"/user/user/triangle-ray/output-log-{dataset_name[:-4]}-{timestamp}"

try:
    log_content = "\n".join(log_messages)
    
    hdfs_fs = fs.HadoopFileSystem.from_uri("hdfs://okeanos-master:54310")
    with hdfs_fs.open_output_stream(log_path) as f:
        f.write(log_content.encode('utf-8'))
    
    print(f"Log saved to HDFS: {log_path}")
except Exception as e:
    print(f"Error saving logs to HDFS: {str(e)}")


time.sleep(2)
ray.shutdown()
