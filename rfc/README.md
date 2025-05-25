To run rfc-spark.py:
spark-submit --master yarn --deploy-mode cluster rfc-spark.py <dataset_index> <executor_instances>

To run rfc-ray.py:
python3 rfc-ray.py <dataset_index> <num_chunks>
