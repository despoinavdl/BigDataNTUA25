# spark-submit --master yarn --deploy-mode cluster k-means-spark.py <dataset_index> <executor_instances> <executor_cores> <executor_memory> <executor_memoryOverhead>
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType, StringType, IntegerType
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler, StandardScaler
import time
import sys

log_messages = []

def log_to_buffer(message):
    log_messages.append(message)

try:
    partitions = 16
    if len(sys.argv) < 3:
            print("Usage: spark-submit --master yarn --deploy-mode cluster k-means-spark.py <dataset_index> <executor_instances>")
            sys.exit(1)

    executor_instances = sys.argv[2]
    

    spark = SparkSession.builder.getOrCreate()

    # HDFS file paths
    index = sys.argv[1]
    data_path = f"hdfs://okeanos-master:54310/datasets/dummy-data-{index}.csv"

    # Load dataset from HDFS
    schema = StructType([
        StructField("num_feature_1", DoubleType(), True),
        StructField("num_feature_2", DoubleType(), True),
        StructField("num_feature_3", DoubleType(), True),
        StructField("category_1", StringType(), True),
        StructField("category_2", StringType(), True),
        StructField("label", IntegerType(), True)
    ])

    load_start = time.time()
    data = spark.read.csv(data_path, header=True, schema=schema)
    load_time = time.time() - load_start

    # Convert data to feature vectors using VectorAssembler
    preprocessing_start = time.time()
    data = data.repartition(partitions)
    numeric_cols = ["num_feature_1", "num_feature_2", "num_feature_3"]
    assembler = VectorAssembler(inputCols=numeric_cols, outputCol="features")
    feature_data = assembler.transform(data).select("features")

    # Apply StandardScaler
    scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withMean=True, withStd=True)
    scaler_model = scaler.fit(feature_data)
    scaled_data = scaler_model.transform(feature_data).select("scaled_features")

    preprocessing_time = time.time() - preprocessing_start

    # Train K-Means model
    train_start = time.time()
    kmeans = KMeans() \
        .setK(2) \
        .setMaxIter(3) \
        .setFeaturesCol("scaled_features") \
        .setPredictionCol("prediction") \
        .setSeed(42) \
        .setInitMode("k-means||")

    model = kmeans.fit(scaled_data)
    train_time = time.time() - train_start

    op_start = time.time()
    # Get cluster centers
    centers = model.clusterCenters()
    # print("\nCluster Centers:")
    # for i, center in enumerate(centers):
    #     print(f"Cluster {i}: {center}")

    # Make predictions on the dataset
    predictions = model.transform(scaled_data)
    cluster_sizes = predictions.groupBy("prediction").count().orderBy("prediction")
    # print("\nCluster Sizes:")
    cluster_sizes.show()

    op_time = time.time() - op_start

    total_time = load_time + preprocessing_time + train_time + op_time 

    log_to_buffer(f"Dataset: dummy-data-{index}.csv, executor_instances={sys.argv[2]}") # executor_cores={sys.argv[3]} executor_memory={sys.argv[4]} executor_memoryOverhead={sys.argv[5]}")
    log_to_buffer(f"Loading Time: {load_time:.2f} seconds")
    log_to_buffer(f"Dataset pre-processing time: {preprocessing_time:.2f} seconds")
    log_to_buffer(f"Training Time: {train_time:.2f} seconds")
    log_to_buffer(f"Operation Time: {op_time:.2f} seconds")
    log_to_buffer(f"Total Time: {total_time:.2f} seconds\n\n")

    folder_path = '/user/user/kmeans-spark/'
    timestamp = time.strftime("%H%M%S")
    log_path = f"{folder_path}output-log-dummy-data-{index}-{timestamp}"
    spark.sparkContext.parallelize(log_messages, 1).saveAsTextFile(log_path)


    # Stop Spark Session
    spark.stop()

except Exception as e:
    import traceback
    log_to_buffer(f"ERROR OCCURRED: {str(e)}")
    folder_path = '/user/user/kmeans-spark/'
    timestamp = time.strftime("%H%M%S")
    index = sys.argv[1]
    log_path = f"{folder_path}output-error-dummy-data-{index}-{timestamp}"
    spark.sparkContext.parallelize(log_messages, 1).saveAsTextFile(log_path)

    traceback.print_exc()
    sys.exit(1)
