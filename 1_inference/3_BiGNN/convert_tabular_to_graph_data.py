# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT:w

import pandas as pd
import glob as glob
import numpy as np
from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser() 
parser.add_argument( "--role1", "-r1", nargs='+', type=str, default="f_15")
parser.add_argument( "--role2", "-r2", nargs='+', type=str, default="f_15")
parser.add_argument( "--normalize", "-n", nargs='+', type=bool, default=False)
args = parser.parse_args()
role1 = args.role1
role2 = args.role2
normalize = args.normalize
print("role1 = ", role1)
print("role2 = ", role2)

# Read Data 

def read_data():
    all_train_csvs = glob.glob("../../data/train/*.csv")
    train_df = []
    for i,file in tqdm(enumerate(all_train_csvs), total=len(all_train_csvs), desc="Read Train files"):
        df_i = pd.read_table(file, engine="pyarrow")
        train_df.append(df_i)
    train_df = pd.concat(train_df, axis=0, ignore_index=True)

    all_test_csvs = glob.glob("../../data/test/*.csv")
    test_df = []
    for i, file in tqdm(enumerate(all_test_csvs), total=len(all_test_csvs), desc="Read Test files"):
        df_i = pd.read_table(file, engine="pyarrow")
        test_df.append(df_i)
    test_df = pd.concat(test_df, axis=0, ignore_index=True)

    return train_df, test_df

train_df, test_df = read_data()

print("processing...")

val_df = train_df[train_df["f_1"]==66]
train_df = train_df[train_df["f_1"]<66]

train_df["train_mask"] = np.ones(len(train_df))
train_df["val_mask"] = np.zeros(len(train_df))
train_df["test_mask"] = np.zeros(len(train_df))
val_df["train_mask"] = np.zeros(len(val_df))
val_df["val_mask"] = np.ones(len(val_df))
val_df["test_mask"] = np.zeros(len(val_df))
test_df["train_mask"] = np.zeros(len(test_df))
test_df["val_mask"] = np.zeros(len(test_df))
test_df["test_mask"] = np.ones(len(test_df))

fullds_df = pd.concat([train_df, val_df, test_df])



date_feature_inds = np.arange(1)
date_features = np.char.add("f_", date_feature_inds.astype(str))

categorical_feature_inds = np.arange(2,42)
categorical_features = np.char.add("f_", categorical_feature_inds.astype(str))

numerical_feature_inds = np.arange(42,80)
numerical_features = np.char.add("f_", numerical_feature_inds.astype(str))

all_feature_inds =np.arange(1,80)
all_features = np.char.add("f_", all_feature_inds.astype(str))

labels = ['is_clicked', 'is_installed']



fullds_df = fullds_df.replace('', np.nan)
for col in categorical_features:
    if (fullds_df[col].isnull().values.any()):
        fullds_df[col] = fullds_df[col].fillna(-1)
        # print(col, fullds_df[col].isnull().sum())
for col in labels:
    if (fullds_df[col].isnull().values.any()):
        fullds_df[col] = fullds_df[col].fillna(-1)
        # print(col, fullds_df[col].isnull().sum())
for col in numerical_features:
    if (fullds_df[col].isnull().values.any()):
        fullds_df[col] = fullds_df[col].replace(np.NaN, fullds_df[col].mean()) 
        # print(col, fullds_df[col].isnull().sum())
if normalize:
    print("WARNING: Using normalization of numerical features")
    fullds_df[numerical_features]=(fullds_df[numerical_features]-fullds_df[numerical_features].mean())/fullds_df[numerical_features].std()

labels = ['is_clicked', 'is_installed']
role1.sort()
role2.sort()
role3 = list(set(fullds_df.columns) - set(role1) - set(role2) - set(labels) )
role3.sort()


role_1_df = fullds_df.drop_duplicates(subset=role1).reset_index()[role1]
role_2_df = fullds_df.drop_duplicates(subset=role2).reset_index()[role2]

def get_group_idx(df, group_id, f= []):
    k = [i for i in df.columns if i not in f][0]
    ret = df.groupby(by = f, as_index = False, sort=False)['f_0'].count().drop('f_0', axis=1)
    ret[f'group_{group_id}'] = ret.index
    ret2 = df.merge(ret, on=f)
    return ret2
trial_df = get_group_idx(fullds_df, 1, role1)
trial_df = get_group_idx(trial_df, 2, role2)
trial_df.rename(columns={"group_1": "src_id", "group_2": "dst_id"}, inplace=True)
filename = './graph_data/sym_recsys_hetero_CSVDatasets/edges.csv.gz'
# filename = './graph_data/sym_recsys_hetero_CSVDatasets/edges'+'_'.join(role1)+'_'+'_'.join(role2) + '.csv.gz'
# trial_df[['src_id', 'dst_id'] + role1 + role2 + role3 + labels].to_csv(filename, compression='gzip', index=False)


chunks = np.array_split(trial_df.index, 100) # split into 100 chunks

for chunck, subset in enumerate(tqdm(chunks, desc='Writing csv file')):
    if chunck == 0: # first row
        trial_df[['src_id', 'dst_id'] + role1 + role2 + role3 + labels].loc[subset].to_csv(filename, mode='w', compression='gzip', index=True)
    else:
        trial_df[['src_id', 'dst_id'] + role1 + role2 + role3 + labels].loc[subset].to_csv(filename, header=None, mode='a', compression='gzip', index=True)

with open('./yaml_store/recsys2graph.yaml', 'a') as f:
    f.write('edge_features: ' + str(list(set(role3) -set(['f_7', 'train_mask', 'test_mask', 'val_mask']))) + '\n')
    f.write('node_features:' + '\n')
    role1.append('f_7')
    role2.append('f_7')
    f.write(' - ' + str(role1) + '\n')
    f.write(' - ' + str(role2) + '\n')