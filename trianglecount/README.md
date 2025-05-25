To run triangle-spark.py: 
spark-submit \
  --master yarn \
  --deploy-mode cluster \
  --name TriangleSpark \
--py-files graphframes.zip \
  --num-executors <executor_instances> \
  --executor-cores 3 \
  --executor-memory 4G \
  --conf spark.scheduler.mode=FAIR \
  --conf spark.scheduler.allocation.mode=FAIR \
  --conf spark.yarn.am.waitTime=10s \
  --conf spark.deploy.spreadOut=true \
  --conf spark.memory.fraction=0.8 \
  --conf spark.memory.storageFraction=0.3 \
  --conf spark.storage.level=MEMORY_AND_DISK \
  triangle-spark.py <dataset_name> <executor_instances>

To run triangle-ray.py: python3 triangle-ray.py <dataset_index> <num_chunks>
