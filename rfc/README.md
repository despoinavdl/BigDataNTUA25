To run rfc-spark.py:
spark-submit \
  --master yarn \
  --deploy-mode cluster \
  --name RFCSpark \
  --num-executors 2 \
  --executor-cores 3\
  --executor-memory 4G \
  --conf spark.scheduler.mode=FAIR \
  --conf spark.scheduler.allocation.mode=FAIR \
  --conf spark.yarn.am.waitTime=10s \
  --conf spark.deploy.spreadOut=true \
  --conf spark.memory.fraction=0.8 \
  --conf spark.memory.storageFraction=0.3 \
  --conf spark.storage.level=MEMORY_AND_DISK \
  rfc-spark.py <dataset_index> <executor_instances>

To run rfc-ray.py:
python3 rfc-ray.py <dataset_index> <num_chunks>
