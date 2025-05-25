# BigDataNTUA25
To start HDFS and YARN:\
start-dfs.sh && start-yarn.sh

To start ray execute on the head node: \
CLASSPATH=$HADOOP_HOME/bin/hdfs classpath --glob ray start --head --port=6379 --num-cpus=4 --object-store-memory=2000000000

And on the worker node: \
CLASSPATH=$HADOOP_HOME/bin/hdfs classpath --glob ray start --address='192.168.0.1:6379' --num-cpus=4 --object-store-memory=2000000000
