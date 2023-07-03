import pandas as pd
import numpy as np
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 150)

import os
from tqdm import tqdm

##STEP 1: Load train and test data into pandas dataframes

def load_files_to_df(filedir):
    files = sorted(os.listdir(filedir))
    df = pd.read_csv(filedir + files[0], sep = '\t')
    for file in tqdm(files[1:]):
        part = pd.read_csv(filedir + file, sep = '\t')
        df = pd.concat([df,part],axis=0)
    return df
train_df = load_files_to_df("../../data/sharechat_recsys2023_data/train/")
test_df = load_files_to_df("../../data/sharechat_recsys2023_data/test/")

print('done loading files')

def fix_na(df):
    df = df.fillna(0)
    for col in df.select_dtypes([float]):
        v = col
        if np.array_equal(df[v], df[v].astype(int)):
            df[v] = df[v].astype(int, copy = False)
    return df
train_df = fix_na(train_df)
test_df = fix_na(test_df)

train_test_df = pd.concat([train_df,test_df])
print(train_test_df.shape)

#STEP 2: SIMILARITY of numerical features using faiss

#path to store similarity distances and indices
data_path="../../data/4_simGNN/similarity"
if not os.path.exists(data_path):
    os.makedirs(data_path)

#subset of numerical features to be used in calculating similarity
feature_id_list = [43, 51, 58, 59, 64, 65, 66, 67, 68, 69, 70,54, 55, 56, 57, 60, 61, 62, 63]
train_columns = [f"f_{i}" for i in feature_id_list]

#min-max normalizing with train_df
train_test_df[train_columns] = (train_test_df[train_columns] - train_df[train_columns].min()) / (train_df[train_columns].max()-train_df[train_columns].min())


# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#https://github.com/facebookresearch/faiss/blob/main/tutorial/python/5-Multiple-GPUs.py


d = len(train_columns)# + train_columns_skew)                           # dimension

train_test_arr=train_test_df[train_columns].to_numpy()

import faiss                     # make faiss available

ngpus = faiss.get_num_gpus()

print("number of GPUs:", ngpus)

cpu_index = faiss.IndexFlatL2(d)

gpu_index = faiss.index_cpu_to_all_gpus(  # build the index
    cpu_index
)

gpu_index.add(train_test_arr)              # add vectors to the index
print(gpu_index.ntotal)

k = 100                          # we want to see 100 nearest neighbors
D, I = gpu_index.search(train_test_arr, k) # actual search
print(I[:5])                   # neighbors of the 5 first queries
print(I[-5:])                  # neighbors of the 5 last queries

np.save(os.path.join(data_path,"full_day45_67_sim100_distances_19num.npy"), D)
np.save(os.path.join(data_path,"full_day45_67_sim100_indices_19num.npy"), I)
print(D.shape)
print(I.shape)


##STEP3: similarity of categorical feature using knn
##takes >1 day to run on 32c CPU
feature_id_list = [2,19,20,21,22,23,24,25,26,27,28,29,33,34,35,36,37,38,39,40,41]
train_columns_cat = [f"f_{i}" for i in feature_id_list]

from sklearn.neighbors import NearestNeighbors
knn = NearestNeighbors(metric='hamming')
knn.fit(train_test_df[train_columns_cat].values)
print("done with fit")


# for train
distances_cat, indices_cat = knn.kneighbors(train_test_df[train_columns_cat].values, n_neighbors=100)
np.save(os.path.join(data_path,"full_day45_67_sim100_distances_cat.npy"), distances_cat)
np.save(os.path.join(data_path,"full_day45_67_sim100_indices_cat.npy"), indices_cat)
