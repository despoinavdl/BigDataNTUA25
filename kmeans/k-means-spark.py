from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType, StringType, IntegerType
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
import time

# Initialize Spark Session with cluster configuration
spark = SparkSession.builder \
    .appName("KMeansClusteringML") \
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
index = "0"
data_path = f"hdfs://okeanos-master:54310/datasets/dummy-data-{index}.csv"
save_path = f"hdfs://okeanos-master:54310/kmeans/kmeans-model-spark-{index}"

# Start timing
start_time = time.time()

# Load dataset from HDFS
schema = StructType([
    StructField("num_feature_1", DoubleType(), True),
    StructField("num_feature_2", DoubleType(), True),
    StructField("num_feature_3", DoubleType(), True),
    StructField("category_1", StringType(), True),
    StructField("category_2", StringType(), True),
    StructField("label", IntegerType(), True)
])

data = spark.read.csv(data_path, header=True, schema=schema)
data = data.repartition(2)
load_time = time.time() - start_time
# "Registering the DataFrame time" would be more accurate
print(f"Data loading time: {load_time:.2f} seconds")

start_time = time.time()
data.count()  # Forces Spark to actually load the data
print(f"Actual data loading time: {time.time() - start_time:.2f} seconds")

# After loading data DEBUG
print(f"Number of partitions: {data.rdd.getNumPartitions()}")
print(f"Schema: {data.schema}")
data.describe().show()  # Basic statistics

# Memory monitoring DEBUG
#print(f"Storage memory used: {spark.sparkContext._jvm.org.apache.spark.storage.StorageUtils.reportStorageUsage()}")

# Convert data to feature vectors using VectorAssembler
start_time = time.time()
numeric_cols = ["num_feature_1", "num_feature_2", "num_feature_3"]
assembler = VectorAssembler(inputCols=numeric_cols, outputCol="features")
feature_data = assembler.transform(data).select("features")
feature_data.cache()
preprocess_time = time.time() - start_time
print(f"Data preprocessing time: {preprocess_time:.2f} seconds")

# Train K-Means model
start_time = time.time()
kmeans = KMeans() \
    .setK(10) \
    .setMaxIter(20) \
    .setFeaturesCol("features") \
    .setPredictionCol("prediction") \
    .setSeed(1) \
    .setInitMode("k-means||")

model = kmeans.fit(feature_data)
train_time = time.time() - start_time
print(f"Model training time: {train_time:.2f} seconds")

# Evaluate model
start_time = time.time()
wssse = model.summary.trainingCost
eval_time = time.time() - start_time
print(f"Model evaluation time: {eval_time:.2f} seconds")
print(f"Within Set Sum of Squared Errors (WSSSE): {wssse:.2f}")

# Get cluster centers
centers = model.clusterCenters()
print("\nCluster Centers:")
for i, center in enumerate(centers):
    print(f"Cluster {i}: {center}")

# Make predictions on the dataset
predictions = model.transform(feature_data)
cluster_sizes = predictions.groupBy("prediction").count().orderBy("prediction")
print("\nCluster Sizes:")
cluster_sizes.show()

# Save model to HDFS
start_time = time.time()
model.write().overwrite().save(save_path)
save_time = time.time() - start_time
print(f"Model saving time: {save_time:.2f} seconds")

# Stop Spark Session
spark.stop()
