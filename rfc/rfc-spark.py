from pyspark.ml.classification import RandomForestClassifier
from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.sql.functions import col, length
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
import os
import sys
import time

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

if(len(sys.argv) != 3):
    print("Usage: python3 rfc-spark.py <dataset_index> <num_executors>")
    exit(1)

# Initialize Spark
spark = SparkSession \
    .builder \
    .appName("RandomForestClassifierSpark") \
    .master("yarn") \
    .config("spark.executor.instances", sys.argv[2])\
    .config("spark.executor.cores", "4") \
    .getOrCreate() 

sc = spark.sparkContext

# print("\n=== CLUSTER INFO ===")
# print(f"Available nodes: {sc.getConf().get('spark.executor.instances')}")
# print(f"Cores per executor: {sc.getConf().get('spark.executor.cores')}")
# print(f"Active nodes: {len(sc._jsc.sc().statusTracker().getExecutorInfos()) - 1}")  # -1 to exclude driver
# print(f"Default parallelism: {sc.defaultParallelism}")
# print("===================\n")

load_start = time.time()
index = sys.argv[1]
df = spark.read.csv(f"hdfs://okeanos-master:54310/datasets/dummy-data-{index}.csv", header=True, inferSchema=True)
load_time = time.time() - load_start

# rows_before = df.count()

df = df.withColumn(
    "new_feature", 
    col("num_feature_1") + length(col("category_1"))
).filter(col("new_feature") > 5)
 
# rows_after = df.count()

preprocessing_start = time.time()

df = df.select(df.num_feature_1, df.num_feature_2, df.num_feature_3, df.new_feature, df.label)

assembler = VectorAssembler(inputCols=["num_feature_1", "num_feature_2", "num_feature_3", "new_feature"], outputCol="input")
scaler = MinMaxScaler(inputCol="input", outputCol="features")
pipeline = Pipeline(stages=[assembler, scaler])
scalerModel = pipeline.fit(df)
scaledData = scalerModel.transform(df).select(["features", "label"])

trainingData, testData = scaledData.randomSplit(weights= [0.7,0.3], seed=100)

preprocessing_time = time.time() - preprocessing_start

rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=5)

# Chain indexers and forest in a Pipeline
pipeline = Pipeline(stages=[rf])

train_start = time.time()
# Train 
model = pipeline.fit(trainingData)
train_time = time.time() - train_start

# Predictions
predictions = model.transform(testData)

total_time = load_time + preprocessing_time + train_time

# Stop Spark
sc.stop()

# print(f"Number of rows before filtering: {rows_before}")
# print(f"Number of rows after filtering: {rows_after}")
print(f"Loading Time: {load_time}")
print(f"Preproccessing Time: {preprocessing_time}")
print(f"Training Time: {train_time}")
print(f"Total Time: {total_time}")
