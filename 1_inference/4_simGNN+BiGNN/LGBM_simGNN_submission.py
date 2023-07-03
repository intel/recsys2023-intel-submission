#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import lightgbm as lgb
import glob as glob
from tqdm import tqdm
import os
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 150)

def merge_datasets(dataset, embedding, feats_to_drop=None):
    copy_fullds_df = dataset
    copy_emb_df = embedding
    reset_ind_copy_fullds_df = copy_fullds_df.reset_index(drop=True)
    reset_ind_copy_emb_df = copy_emb_df.reset_index(drop=True).drop(columns=feats_to_drop)
    rearranged_df = reset_ind_copy_fullds_df.merge(reset_ind_copy_emb_df, on='f_0', how='left', suffixes=('_1', '_2'))
    return rearranged_df

def read_data_graph(file):
    print(file)
    df = pd.read_csv(file, compression='gzip', engine="pyarrow")
    return df
print("Reading embeddings...")

## STEP 1: Load embeddings from 3_BiGNN (Bipartite GNN self-supervised task)
file = "../3_BiGNN/best_embedding/gnn_emb_is_clicked.csv.gz"
save_file = file + ".xgb.model"
click_emb_fullds_df = read_data_graph(file=file)

file = "../3_BiGNN/best_embedding/gnn_emb_is_installed.csv.gz"
save_file = file + ".xgb.model"
install_emb_fullds_df = read_data_graph(file=file)


def read_parquet_data(file):
    return pd.read_parquet(file, engine='pyarrow')

print("Reading FE...")

## STEP 2: Load feature engineering from 1_LearningFE 
file_train = "../../data/1_LearningFE/train_processed.parquet" 
file_test = "../../data/1_LearningFE/test_processed.parquet"
train_fe_df = read_parquet_data(file_train)
test_fe_df = read_parquet_data(file_test)

experiment_tag = "Exp1_lr01_iter5000"
selected_features = ['dow']
selected_features += [f"f_{i}" for i in range(0,80)]
selected_features += [f"f_{i}_CE" for i in [2,4,6,13,15,18] + [78,75,50,20,24]]
selected_features += [f"f_{i}_idx" for i in range(2,23) if i not in [2,4,6,15]]
excluded_features = ["is_clicked", "is_installed"]
print(selected_features)
all_features = selected_features 
print('Done!')

## STEP 3: Load embeddings from similarity GNN
## NOTE: here the best expriment path is hardcoded -  CHANGE ACCORDING TO YOUR 10 SIMILATIONS
#file = "/media/SSD4T/rbrugaro/Recys2023/Exp21_noTestTest-2023-06-14-18-40/recsys_data45_67_w_gnn_emb_run_7_noTestTest.parquet"
file = "../../0_train/4_simGNN/Exp1-2023-06-27-22-16/recsys_data45_67_w_gnn_emb_run_1.parquet"

full_similarity_df = read_parquet_data(file)

orig_features = ['is_installed', 'is_clicked']
for i in np.arange(1,80):
    orig_features.append('f_' + str(i))

#simGNN embedding columns names are just a number from 0-255
for i in np.arange(0,256):
    all_features = all_features + [str(i)]

## STEP 4: Combine all dataset for LGBM input in the following order: LearningFE - simGNN - BiGNNisclick - BiGNNisinstall
print("Merge datasets...")
train_df = merge_datasets(train_fe_df, full_similarity_df[full_similarity_df['f_1']<67], orig_features)
test_df = merge_datasets(test_fe_df, full_similarity_df[full_similarity_df['f_1']==67], orig_features)
train_df = merge_datasets(train_df, click_emb_fullds_df[click_emb_fullds_df['f_1']<67].loc[:,click_emb_fullds_df.columns!=''], orig_features)
test_df = merge_datasets(test_df, click_emb_fullds_df[click_emb_fullds_df['f_1']==67].loc[:,click_emb_fullds_df.columns!=''], orig_features)
train_df = merge_datasets(train_df, install_emb_fullds_df[install_emb_fullds_df['f_1']<67].loc[:,click_emb_fullds_df.columns!=''], orig_features)
test_df = merge_datasets(test_df, install_emb_fullds_df[install_emb_fullds_df['f_1']==67].loc[:,click_emb_fullds_df.columns!=''], orig_features)

print('Done!')

df = train_df

val_df = df[df["f_1"]==66]
train_df = df[df["f_1"]<67]

all_features.remove('f_0')
all_features.remove('f_1')
all_features.remove('f_7')

#BiGNN embedding column names (i,e: n0_e10_1, n0_e10_2...)
emb_features = []
for node in np.arange(2):
    for dimension in np.arange(64):
        emb_features.append('n' + str(node) + '_e' + str(dimension) + '_1')
        emb_features.append('n' + str(node) + '_e' + str(dimension) + '_2')
all_features_after_embedding = all_features + emb_features

## Load pretrained LGBM model
model = lgb.Booster(model_file = os.path.join("../../data/4_simGNN",  "lgbm_trained_simGNN_BiGNN.mdl"))

is_installed_pred = model.predict(test_df[all_features_after_embedding])
is_clicked_pred = model.predict(test_df[all_features_after_embedding])
submission_df = pd.DataFrame(is_installed_pred, columns=['is_installed'])
submission_df['is_clicked'] = is_clicked_pred
submission_df['RowId'] = test_df['f_0']
submission_df.to_csv('../../data/submissions/lgbm_with_simGNN_BiGNN_emb_' + experiment_tag + '.csv', index=False, sep="\t")
