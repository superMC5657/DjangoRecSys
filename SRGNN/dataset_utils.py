"""
Updated on Dec 20, 2020

create implicit ml-1m dataset

@author: Ziyao Geng(zggzy1996@163.com)
"""
import time

import pandas as pd
import numpy as np
import random
from tqdm import tqdm
from collections import defaultdict


def sparseFeature(feat, feat_num, embed_dim=4):
    """
    create dictionary for sparse feature
    :param feat: feature name
    :param feat_num: the total number of sparse features that do not repeat
    :param embed_dim: embedding dimension
    :return:
    """
    return {'feat': feat, 'feat_num': feat_num, 'embed_dim': embed_dim}


def create_ml_1m_dataset(sample_file, user_file, trans_score=2, embed_dim=8, test_neg_num=100):
    """
    :param sample_file: A string. dataset path.
    :param user_file: A string. dataset path.
    :param trans_score: A scalar. Greater than it is 1, and less than it is 0.
    :param embed_dim: A scalar. latent factor.
    :param test_neg_num: A scalar. The number of test negative samples
    :return: user_num, item_num, train_df, test_df
    """
    print('==========Data Preprocess Start=============')
    data_df = pd.read_csv(sample_file, sep="::", engine='python',
                          names=['user_id', 'item_id', 'label', 'timestamp'])

    user_dict = read_user_df(user_file)

    # filtering
    data_df['item_count'] = data_df.groupby('item_id')['item_id'].transform('count')
    data_df = data_df[data_df.item_count >= 5]
    # trans score
    data_df = data_df[data_df.label >= trans_score]
    # sort
    data_df["timestamp"] = data_df["timestamp"].apply(get_time_tag)
    data_df = data_df.sort_values(by=['user_id', 'timestamp'])
    # split dataset and negative sampling
    print('============Negative Sampling===============')
    train_data, val_data, test_data = defaultdict(list), defaultdict(list), defaultdict(list)
    item_id_max = data_df['item_id'].max()
    for user_id_timestamp, df in tqdm(
            data_df[['user_id', 'item_id', "timestamp"]].groupby(by=['user_id'])):
        pos_list = df['item_id'].tolist()
        if len(pos_list) <= 0:
            return
        user_id = user_id_timestamp

        def gen_neg():
            neg = pos_list[0]
            while neg in set(pos_list):
                neg = random.randint(1, item_id_max)
            return neg

        neg_list = [gen_neg() for i in range(len(pos_list) + test_neg_num)]
        gender, age, occupation = get_user_info(user_id, user_dict)
        for i in range(1, len(pos_list)):
            hist_i = pos_list[:i]
            if i == len(pos_list) - 1:
                test_data['user_id'].append(user_id)
                test_data['pos_id'].append(pos_list[i])
                test_data['neg_id'].append(neg_list[i:])
                test_data['gender'].append(gender)
                test_data['age'].append(age)
                test_data['occupation'].append(occupation)
            elif i == len(pos_list) - 2:
                val_data['user_id'].append(user_id)
                val_data['pos_id'].append(pos_list[i])
                val_data['neg_id'].append(neg_list[i])
                val_data['gender'].append(gender)
                val_data['age'].append(age)
                val_data['occupation'].append(occupation)
            else:
                train_data['user_id'].append(user_id)
                train_data['pos_id'].append(pos_list[i])
                train_data['neg_id'].append(neg_list[i])
                train_data['gender'].append(gender)
                train_data['age'].append(age)
                train_data['occupation'].append(occupation)
    # feature columns
    user_num, item_num = data_df['user_id'].max() + 1, data_df['item_id'].max() + 1
    item_feat_col = [sparseFeature('user_id', user_num, embed_dim),
                     sparseFeature('item_id', item_num, embed_dim),
                     sparseFeature('gender', 2, embed_dim),
                     sparseFeature('age', 7, embed_dim),
                     sparseFeature('occupation', 21, embed_dim)]
    # shuffle
    random.shuffle(train_data)
    random.shuffle(val_data)
    train = [np.array(train_data['user_id']), np.array(train_data['pos_id']),
             np.array(train_data['neg_id']), np.array(train_data['gender']),
             np.array(train_data['age']), np.array(train_data['occupation'])]
    val = [np.array(val_data['user_id']), np.array(val_data['pos_id']),
           np.array(val_data['neg_id']), np.array(val_data['gender']), np.array(val_data['age']),
           np.array(val_data['occupation'])]
    test = [np.array(test_data['user_id']), np.array(test_data['pos_id']),
            np.array(test_data['neg_id']), np.array(test_data['gender']), np.array(test_data['age']),
            np.array(test_data['occupation'])]
    print('============Data Preprocess End=============')
    return item_feat_col, train, val, test


# create_ml_1m_dataset('../dataset/ml-1m/ratings.dat')

def read_user_df(file):
    data_df = pd.read_csv(file, sep="::", engine='python',
                          names=['user_id', 'gender', 'age', 'occupation', 'zipcode'])

    age_dict = {1: 0, 18: 1, 25: 2, 35: 3, 45: 4, 50: 5, 56: 6}

    def convert_gender(x):
        if x == "M":
            return 0
        else:
            return 1

    def convert_age(x):
        return age_dict[x]

    data_df["gender"] = data_df["gender"].map(convert_gender)
    data_df["age"] = data_df["age"].map(convert_age)
    data_df = data_df.set_index("user_id")
    return data_df.to_dict("index")


def get_user_info(user_id, user_dict):
    """
    :param user_id: A scalar. user id
    :param user_dict: A dict. user info
    :return: user_info
    """
    user_info = user_dict[user_id]

    return user_info['gender'], user_info['age'], user_info['occupation']


def get_time_tag(timestamp):
    hour = time.localtime(int(timestamp)).tm_hour
    if hour <= 6:
        return 0
    elif hour <= 12:
        return 1
    elif hour <= 18:
        return 2
    else:
        return 3


if __name__ == '__main__':
    user_df = read_user_df(file='SRGNN/datasets/ml-1m/users.dat')
    feature_columns, train, val, test = create_ml_1m_dataset("SRGNN/datasets/ml-1m/ratings.dat", "SRGNN/datasets/ml-1m/users.dat")
    print(train)
    info = get_user_info(user_id=1, user_dict=user_df)
    print(info)
