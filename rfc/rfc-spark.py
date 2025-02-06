from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType, StringType, IntegerType
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import time

# Step 1: Initialize Spark Session
start_time = time.time()
spark = SparkSession.builder \
    .appName("RandomForestClassifierTraining") \
    .master("yarn") \
    .config("spark.executor.instances", "2") \
    .config("spark.executor.cores", "4") \
    .config("spark.executor.memory", "4g") \
    .config("spark.dynamicAllocation.enabled", "false") \
    .config("spark.eventLog.enabled", "true") \
    .config("spark.eventLog.dir", "hdfs://okeanos-master:54310/spark.eventLog") \
    .config("spark.executor.memoryOverhead", "2g") \
    .getOrCreate()
init_time = time.time() - start_time
print(f"Time taken to initialize Spark session: {init_time:.2f} seconds")

# Step 2: Define the schema
schema = StructType([
    StructField("num_feature_1", DoubleType(), True),
    StructField("num_feature_2", DoubleType(), True),
    StructField("num_feature_3", DoubleType(), True),
    StructField("category_1", StringType(), True),
    StructField("category_2", StringType(), True),
    StructField("label", IntegerType(), True)
])

# Step 3: Read the dataset from HDFS
start_time = time.time()
index = "1"
hdfs_path = f"hdfs://okeanos-master:54310/datasets/dummy-data-{index}.csv"
save_path = f"hdfs://okeanos-master:54310/rfc/rfc-model-spark-{index}"
df = spark.read.csv(hdfs_path, schema=schema, header=True)
load_time = time.time() - start_time
print(f"Time taken to load data from HDFS: {load_time:.2f} seconds")

# Step 4: Preprocess the data
start_time = time.time()
# Index categorical columns
indexer_category_1 = StringIndexer(inputCol="category_1", outputCol="category_1_index")
indexer_category_2 = StringIndexer(inputCol="category_2", outputCol="category_2_index")

# One-hot encode categorical columns
encoder_category_1 = OneHotEncoder(inputCol="category_1_index", outputCol="category_1_encoded")
encoder_category_2 = OneHotEncoder(inputCol="category_2_index", outputCol="category_2_encoded")

# Assemble all features into a single vector column
feature_columns = ["num_feature_1", "num_feature_2", "num_feature_3", "category_1_encoded", "category_2_encoded"]
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
preprocess_time = time.time() - start_time
print(f"Time taken to preprocess data: {preprocess_time:.2f} seconds")

# Step 5: Split the data into training and testing sets
start_time = time.time()
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)
split_time = time.time() - start_time
print(f"Time taken to split data: {split_time:.2f} seconds")

# Step 6: Define the Random Forest Classifier
rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=100, maxDepth=10, seed=42)

# Step 7: Create a pipeline
pipeline = Pipeline(stages=[indexer_category_1, indexer_category_2, encoder_category_1, encoder_category_2, assembler, rf])

# Step 8: Train the model
start_time = time.time()
model = pipeline.fit(train_data)
train_time = time.time() - start_time
print(f"Time taken to train the model: {train_time:.2f} seconds")

# Step 9: Make predictions on the test data
start_time = time.time()
predictions = model.transform(test_data)
predict_time = time.time() - start_time
print(f"Time taken to make predictions: {predict_time:.2f} seconds")

# Step 10: Evaluate the model
start_time = time.time()
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
evaluate_time = time.time() - start_time
print(f"Time taken to evaluate the model: {evaluate_time:.2f} seconds")
print(f"Test Accuracy: {accuracy:.4f}")

# Step 11: Save the model (optional)
start_time = time.time()
model.write().overwrite().save(save_path)
save_time = time.time() - start_time
print(f"Time taken to save the model: {save_time:.2f} seconds")

# Step 12: Stop the Spark session
spark.stop()
