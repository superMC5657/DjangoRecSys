# -*- coding: utf-8 -*-
# !@time: 2022/5/31 上午6:51
# !@author: superMC @email: 18758266469@163.com
# !@fileName: inference.py


import time

import numpy as np
import pandas as pd
import sklearn.metrics.pairwise as pw
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import collect_list

from SRGNN.file_util import load_dict

data_path = "SRGNN/datasets/ml-1m/ratings.csv"
item_dict_path = "SRGNN/datasets/ml-1m/item_dict.json"
item_embedding = pd.read_csv("SRGNN/datasets/ml-1m/embedding.csv", header=None)
item_embedding_values = item_embedding.values

item_dict = load_dict(item_dict_path)
item_1 = {}
item_2 = {}
for k, v in item_dict.items():
    item_1[int(k)] = v
    item_2[v] = int(k)

conf = SparkConf().setMaster("local").setAppName("infer")
sc = SparkContext(conf=conf)
sc.setLogLevel("ERROR")
spark = SparkSession(sc)
user_item_dict = {}
df = spark.read.csv(data_path).toDF("session_id", "origin_iid", "rating", "timestamp", "user_id", "item_id")
df = df.groupby("user_id").agg(collect_list("item_id"), collect_list("timestamp")) \
    .toDF("user_id", "item_id_list", "timestamp_list").rdd.collect()
spark.stop()

for line in df[1:]:
    user_id = line['user_id']
    item_id_list: list = line['item_id_list']
    timestamp_list: list = line['timestamp_list']
    item_timestamp_list = list(zip(item_id_list, timestamp_list))
    item_timestamp_list.sort(key=lambda x: x[1], reverse=True)
    new_item_id_list = [x[0] for x in item_timestamp_list]
    user_item_dict[user_id] = item_id_list


def infer(user_id: str, seed_num=5, recall_num=5):
    item_id_list = user_item_dict[user_id]
    total_recall_list = []
    for index in item_id_list[:seed_num]:
        total_recall_list.extend(get_sim(index, recall_num))
    print(total_recall_list)


def get_sim(index, recall_num):
    recall_list = []
    try:
        index = item_1[int(index)]
        id_embedding = item_embedding.iloc[index].values[np.newaxis, :]
        scores = list(np.squeeze(pw.pairwise_distances(id_embedding, item_embedding_values, metric='cosine')))
        index_list = [i for i in range(len(scores))]
        index_scores_list = list(zip(index_list, scores))
        index_scores_list.sort(key=lambda x: x[1], reverse=False)
        index_scores_list = index_scores_list[:recall_num + 1]
        recall_index_list = [x[0] for x in index_scores_list][1:]
        for elem in recall_index_list:
            recall_list.append(item_2[elem])
    except Exception as e:
        print(e)
    return recall_list


if __name__ == '__main__':
    for i in range(10):
        start = time.time()
        infer(str(i))
        print(time.time() - start)
