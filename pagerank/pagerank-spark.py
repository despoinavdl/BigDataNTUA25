from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import split, col
from graphframes import GraphFrame
import sys
import time 
import os

# Ensure a dataset name is provided as a command-line argument
if len(sys.argv) < 2:
    print("Usage: python3 pagerank-spark.py <dataset_name>")
    sys.exit(1)

# Initialize Spark session
spark = SparkSession.builder \
    .appName("PageRankHDFS") \
    .config("spark.jars.packages", "graphframes:graphframes:0.8.2-spark3.0-s_2.12") \
    .config("spark.driver.extraClassPath", f"{os.environ['SPARK_HOME']}/jars/graphframes-0.8.2-spark3.0-s_2.12.jar") \
    .config("spark.executor.extraClassPath", f"{os.environ['SPARK_HOME']}/jars/graphframes-0.8.2-spark3.0-s_2.12.jar") \
    .getOrCreate()

dataset_name = sys.argv[1]
if(dataset_name == "web-Google.txt"):
    sep = "\t"
else:
    sep = " "

# Define HDFS path to CSV file
hdfs_path = "hdfs://okeanos-master:54310/datasets/" + dataset_name
output_hdfs_path = "hdfs://okeanos-master:54310/pagerank/pagerank-spark-results-" + dataset_name
times_path = "hdfs://okeanos-master:54310/pagerank/pagerank-spark-times-" + dataset_name

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
edges_df.show()
vertices_df.show()

# Create a GraphFrame
graph = GraphFrame(vertices_df, edges_df)

process_time = time.time() - process_start_time
print(f"Dataset pre-processing time: {process_time:.2f} seconds")

exec_start_time = time.time()
# Run PageRank
pagerank_results = graph.pageRank(resetProbability=0.15, maxIter=10)

# Stop measuring execution time
exec_time = time.time() - exec_start_time

# Print execution time
print(f"Execution Time: {exec_time:.2f} seconds")

save_start_time = time.time()
# Show the PageRank scores
pagerank_results.vertices.select("id", "pagerank").show()
pagerank_results.vertices.write.mode("overwrite").csv(output_hdfs_path, header=True)

save_time = time.time() - save_start_time
print(f"Save time: {save_time:.2f} seconds")

times_data = [
    Row(metric="Loading Time", value=f"{load_time:.2f} seconds"),
    Row(metric="Dataset pre-processing time", value=f"{process_time:.2f} seconds"),
    Row(metric="Execution Time", value=f"{exec_time:.2f} seconds"),
    Row(metric="Save time", value=f"{save_time:.2f} seconds")
]

times_df = spark.createDataFrame(times_data)
times_df.write.mode("overwrite").csv(times_path, header=True)


# Stop Spark session
spark.stop()
