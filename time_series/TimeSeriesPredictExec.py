#!/usr/bin/env python
# coding=utf-8
# @author 18099099
import os
cmd = 'source change_spark_version spark-2.1.0 && /home/bigdata/software/spark-2.1.0.7-bin-2.4.0.10/bin/spark-submit ' \
      '--master yarn-cluster --num-executors 40 ' \
      '--conf spark.hadoop.mapreduce.output.fileoutputformat.compress=false ' \
      '--py-files redis-2.10.6.zip ' \
      '--executor-memory 4g TimeSeriesPredict.py'
os.system(cmd)