from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType, StringType, IntegerType
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler, StandardScaler
import time
import sys

# Initialize Spark Session with cluster configuration
spark = SparkSession.builder \
    .appName("KMeansSpark") \
    .master("yarn") \
    .config("spark.executor.instances", "2") \
    .config("spark.executor.cores", "4") \
    .config("spark.executor.memory", "4G") \
    .config("spark.dynamicAllocation.enabled", "false") \
    .config("spark.eventLog.enabled", "true") \
    .config("spark.eventLog.dir", "hdfs://okeanos-master:54310/spark.eventLog") \
    .config("spark.executor.memoryOverhead", "2g") \
    .getOrCreate()


# HDFS file paths
index = sys.argv[1]
data_path = f"hdfs://okeanos-master:54310/datasets/dummy-data-{index}.csv"
# save_path = f"hdfs://okeanos-master:54310/kmeans/kmeans-model-spark-{index}"

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
# data = data.repartition(500)
# "Registering the DataFrame time" would be more accurate
# print(f"Data loading time: {load_time:.2f} seconds")/


# start_time = time.time()
data.limit(1).show()  # Forces Spark to actually load the data
# print(f"Actual data loading time: {time.time() - start_time:.2f} seconds")
load_time = time.time() - load_start

# After loading data DEBUG
# print(f"Number of partitions: {data.rdd.getNumPartitions()}")
# print(f"Schema: {data.schema}")
# data.describe().show()  # Basic statistics

# Memory monitoring DEBUG
#print(f"Storage memory used: {spark.sparkContext._jvm.org.apache.spark.storage.StorageUtils.reportStorageUsage()}")

# Convert data to feature vectors using VectorAssembler
preprocessing_start = time.time()
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
# print(f"Model training time: {train_time:.2f} seconds")

# Evaluate model
# start_time = time.time()
# wssse = model.summary.trainingCost
# eval_time = time.time() - start_time
# print(f"Model evaluation time: {eval_time:.2f} seconds")
# print(f"Within Set Sum of Squared Errors (WSSSE): {wssse:.2f}")

# Get cluster centers
centers = model.clusterCenters()
print("\nCluster Centers:")
for i, center in enumerate(centers):
    print(f"Cluster {i}: {center}")

# Make predictions on the dataset
predictions = model.transform(scaled_data)
cluster_sizes = predictions.groupBy("prediction").count().orderBy("prediction")
print("\nCluster Sizes:")
cluster_sizes.show()

# Save model to HDFS
# start_time = time.time()
# model.write().overwrite().save(save_path)
# save_time = time.time() - start_time
# print(f"Model saving time: {save_time:.2f} seconds")

# Stop Spark Session
spark.stop()

total_time = load_time + preprocessing_time + train_time
f = open(f"/home/user/kmeans/results/kmeans-spark-data-{index}.output", "w")
f.write(f"Loading Time: {load_time}\n")
f.write(f"Preproccessing Time: {preprocessing_time}\n")
f.write(f"Training Time: {train_time}\n")
f.write(f"Total Time: {total_time}\n")
f.close()


print(f"Loading Time: {load_time}")
print(f"Preproccessing Time: {preprocessing_time}")
print(f"Training Time: {train_time}")
print(f"Total Time: {total_time}")
