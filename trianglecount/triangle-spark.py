from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import col
from graphframes import GraphFrame
import sys
import time 

# Ensure a dataset name is provided as a command-line argument
if len(sys.argv) < 2:
    print("Usage: python3 triangle-spark.py <dataset_name>")
    print("Available datasets: tmp-graph.txt, musae_DE_edges.csv, web-Google.txt, sx-stackoverflow.txt")
    sys.exit(1)

# Initialize Spark session
spark = SparkSession.builder \
    .appName("TriangleCountHDFS") \
    .config("spark.jars.packages", "graphframes:graphframes:0.8.2-spark3.0-s_2.12") \
    .config("spark.executor.instances", "2") \
    .config("spark.executor.cores", "4") \
    .config("spark.executor.memory", "4G") \
    .config("spark.dynamicAllocation.enabled", "false") \
    .config("spark.eventLog.enabled", "true") \
    .config("spark.eventLog.dir", "hdfs://okeanos-master:54310/spark.eventLog") \
    .config("spark.executor.memoryOverhead", "2g") \
    .getOrCreate()

dataset_name = sys.argv[1]
if dataset_name == "web-Google.txt":
    sep = "\t"
else:
    sep = " "

# Define HDFS path to CSV file
hdfs_path = "hdfs://okeanos-master:54310/datasets/" + dataset_name
# output_hdfs_path = "hdfs://okeanos-master:54310/trianglecount/trianglecount-spark-results-" + dataset_name

# Start measuring execution time
load_start_time = time.time()

# Load the CSV file from HDFS
edges_df = spark.read.load(hdfs_path, format="csv", sep=sep, header="true")

load_time = time.time() - load_start_time
print(f"Loading Time : {load_time:.2f} seconds")

process_start_time = time.time()

# Split the "u v" column into separate columns "src" and "dst"
edges_df = edges_df.withColumn("src", col("u").cast("int")) \
                   .withColumn("dst", col("v").cast("int")) \
                   .drop("u").drop("v")

# Extract unique vertices
vertices_df = edges_df.select("src").union(edges_df.select("dst")).distinct().toDF("id")

# Show the result
# edges_df.show()
# vertices_df.show()

# Create a GraphFrame
graph = GraphFrame(vertices_df, edges_df)

process_time = time.time() - process_start_time
print(f"Dataset pre-processing time: {process_time:.2f} seconds")

exec_start_time = time.time()
# Run Triangle Count
triangle_count_results = graph.triangleCount()

# Stop measuring execution time
exec_time = time.time() - exec_start_time

# Print execution time
print(f"Execution Time: {exec_time:.2f} seconds")

# save_start_time = time.time()
# Show the Triangle Count results
triangle_count_results.select("id", "count").show()

# Save the results
# triangle_count_results.write.mode("overwrite").csv(output_hdfs_path, header=True)
# save_time = time.time() - save_start_time
# print(f"Save time: {save_time:.2f} seconds")

# Stop Spark session
spark.stop()

total_time = load_time + process_time + exec_time
file_path = f"/home/user/trianglecount/results/trianglecount-spark-data-{dataset_name}.output"

with open(file_path, "w") as f:
    f.write(f"Loading Time: {load_time:.2f} seconds\n")
    f.write(f"Dataset pre-processing time: {process_time:.2f} seconds\n")
    f.write(f"Execution Time: {exec_time:.2f} seconds\n")
    # f.write(f"Save Time: {save_time:.2f} seconds\n")
    f.write(f"Total Time: {total_time:.2f} seconds\n")

print(f"Loading Time: {load_time:.2f} seconds")
print(f"Dataset pre-processing time: {process_time:.2f} seconds")
print(f"Execution Time: {exec_time:.2f} seconds")
# print(f"Save Time: {save_time:.2f} seconds")
print(f"Total Time: {total_time:.2f} seconds")
