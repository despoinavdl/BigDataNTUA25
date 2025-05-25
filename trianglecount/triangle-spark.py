# spark-submit --master yarn --deploy-mode cluster --py-files graphframes.zip triangle-spark.py <dataset> <executor_instances>
from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import col
from graphframes import GraphFrame
import sys
import time 
import traceback

log_messages = []

def log_to_buffer(message):
    log_messages.append(message)

try:
    partitions = 16
    if len(sys.argv) < 3:
            print("Usage: spark-submit --master yarn --deploy-mode cluster --py-files graphframes.zip triangle-spark.py <dataset_index> <executor_instances>")
            sys.exit(1)

    executor_instances = sys.argv[2]

    spark = SparkSession.builder.getOrCreate()

    dataset_name = sys.argv[1]
    if dataset_name in ["web-Google.txt"]:
        sep = "\t"
    else:
        sep = " "

    hdfs_path = "hdfs://okeanos-master:54310/datasets/" + dataset_name

    load_start_time = time.time()
    edges_df = spark.read.load(hdfs_path, format="csv", sep=sep, header="true")
    load_time = time.time() - load_start_time

    process_start_time = time.time()
    # Repartition data to improve parallelism
    edges_df = edges_df.repartition(partitions, "u")

    # Split the "u v" column into separate columns "src" and "dst"
    edges_df = edges_df.withColumn("src", col("u").cast("int")) \
                    .withColumn("dst", col("v").cast("int")) \
                    .drop("u").drop("v") \
                    .repartition(partitions, "src", "dst") 

    # Extract unique vertices
    vertices_df = edges_df.select("src").union(edges_df.select("dst")).distinct().toDF("id")

    # Create a GraphFrame
    graph = GraphFrame(vertices_df, edges_df)
    process_time = time.time() - process_start_time

    exec_start_time = time.time()
    # Run Triangle Count
    triangle_count_results = graph.triangleCount()
    exec_time = time.time() - exec_start_time

    op_start = time.time()
    # Collect the results as a list of Row objects
    triangle_count_list = triangle_count_results.collect()
    # Calculate the total number of unique triangles
    total_triangle_count = sum([row['count'] for row in triangle_count_list])
    unique_triangle_count = total_triangle_count / 3
    op_time = time.time() - op_start

    total_time = load_time + process_time + exec_time + op_time
    
    log_to_buffer(f"Dataset: {dataset_name}, executor_instances={sys.argv[2]}") # executor_cores={sys.argv[3]} executor_memory={sys.argv[4]} executor_memoryOverhead={sys.argv[5]}")
    log_to_buffer(f"Loading Time: {load_time:.2f} seconds")
    log_to_buffer(f"Dataset pre-processing time: {process_time:.2f} seconds")
    log_to_buffer(f"Execution Time: {exec_time:.2f} seconds")
    log_to_buffer(f"Operation Time: {op_time:.2f} seconds")
    log_to_buffer(f"Total Time: {total_time:.2f} seconds\n\n")

    log_to_buffer(f"Total number of unique triangles: {unique_triangle_count}")

    folder_path = '/user/user/triangle-spark/'
    timestamp = time.strftime("%H%M%S")
    log_path = f"{folder_path}output-log-{dataset_name[:-4]}-{timestamp}"
    spark.sparkContext.parallelize(log_messages, 1).saveAsTextFile(log_path)

    spark.stop()

except Exception as e:
    import traceback
    log_to_buffer(f"ERROR OCCURRED: {str(e)}")
    folder_path = '/user/user/triangle-spark/'
    timestamp = time.strftime("%H%M%S")
    dataset_name = sys.argv[1]
    log_path = f"{folder_path}output-error-{dataset_name[:-4]}-{timestamp}"
    spark.sparkContext.parallelize(log_messages, 1).saveAsTextFile(log_path)

    traceback.print_exc()
    sys.exit(1)
