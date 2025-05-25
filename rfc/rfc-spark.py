from pyspark import SparkContext
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, length
import os
import sys
import time

log_messages = []

def log_to_buffer(message):
    log_messages.append(message)

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

try: 
    if(len(sys.argv) != 3):
        print("Usage: python3 rfc-spark.py <dataset_index> <num_executors>")
        exit(1)

    spark = SparkSession.builder.getOrCreate() 

    sc = spark.sparkContext

    load_start = time.time()
    index = sys.argv[1]
    df = spark.read.csv(f"hdfs://okeanos-master:54310/datasets/dummy-data-{index}.csv", header=True, inferSchema=True)
    load_time = time.time() - load_start

    # rows_before = df.count()

    df = df.withColumn(
        "new_feature", 
        col("num_feature_1") + length(col("category_1"))
    ).filter(
        (col("new_feature") > 11) & 
        (col("num_feature_2") + col("num_feature_3") < 0.2)
    )
    
    # rows_after = df.count()

    preprocessing_start = time.time()

    df = df.select(df.num_feature_1, df.num_feature_2, df.num_feature_3, df.new_feature, df.label)

    assembler = VectorAssembler(inputCols=["num_feature_1", "num_feature_2", "num_feature_3", "new_feature"], outputCol="input")
    scaler = StandardScaler(inputCol="input", outputCol="features")
    pipeline = Pipeline(stages=[assembler, scaler])
    scalerModel = pipeline.fit(df)
    scaledData = scalerModel.transform(df).select(["features", "label"])

    trainingData, testData = scaledData.randomSplit(weights= [0.7,0.3], seed=42)

    preprocessing_time = time.time() - preprocessing_start

    rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=4, maxDepth=3)

    # Chain indexers and forest in a Pipeline
    pipeline = Pipeline(stages=[rf])

    train_start = time.time()
    # Train 
    model = pipeline.fit(trainingData)
    train_time = time.time() - train_start

    # Predictions
    op_start = time.time()
    predictions = model.transform(testData)
    op_time = time.time() - op_start

    total_time = load_time + preprocessing_time + train_time + op_time

    log_to_buffer(f"Dataset: dummy-data-{index}.csv, executor_instances={sys.argv[2]}") 
    log_to_buffer(f"Loading Time: {load_time:.2f} seconds")
    log_to_buffer(f"Dataset pre-processing time: {preprocessing_time:.2f} seconds")
    log_to_buffer(f"Training Time: {train_time:.2f} seconds")
    log_to_buffer(f"Operation Time: {op_time:.2f} seconds")
    log_to_buffer(f"Total Time: {total_time:.2f} seconds\n\n")

    folder_path = '/user/user/rfc-spark/'
    timestamp = time.strftime("%H%M%S")
    log_path = f"{folder_path}output-log-dummy-data-{index}-{timestamp}"
    spark.sparkContext.parallelize(log_messages, 1).saveAsTextFile(log_path)

    sc.stop()

except Exception as e:
    import traceback
    log_to_buffer(f"ERROR OCCURRED: {str(e)}")
    folder_path = '/user/user/rfc-spark/'
    timestamp = time.strftime("%H%M%S")
    index = sys.argv[1]
    log_path = f"{folder_path}output-error-dummy-data-{index}-{timestamp}"
    spark.sparkContext.parallelize(log_messages, 1).saveAsTextFile(log_path)

    traceback.print_exc()
    sys.exit(1)
