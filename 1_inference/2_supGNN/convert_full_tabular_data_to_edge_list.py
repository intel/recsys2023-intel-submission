import argparse
import glob as glob
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

# step 1 : parse user input
parser = argparse.ArgumentParser() 
parser.add_argument( "--role1", "-r1", nargs='+', type=str, default="f_6")
parser.add_argument( "--role2", "-r2", nargs='+', type=str, default="f_2 f_4 f_16")
parser.add_argument( "--normalize", "-n", nargs='+', type=bool, default=False)
parser.add_argument( "--data_dir", "-d", default="/localdisk/akakne/recsys2023/data/sharechat_recsys2023_data")
parser.add_argument( "--output_file", "-o", default="/localdisk/akakne/recsys2023/data/supGNN_graph_data/full_graph/full_edges.csv.gz")
args = parser.parse_args()
role1 = args.role1.split()
role2 = args.role2.split()
normalize = args.normalize
data_dir = args.data_dir
train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")
output_file = args.output_file
print("role1 = ", role1)
print("role2 = ", role2)

# step 2 : read data 
def read_data(data_dir):
    all_csvs = glob.glob(os.path.join(data_dir, "*.csv"))
    df = []
    for _, file in tqdm(enumerate(all_csvs), total=len(all_csvs), desc="Read files"):
        df_i = pd.read_table(file, engine="pyarrow")
        df.append(df_i)
    return pd.concat(df, axis=0, ignore_index=True)

train_df = read_data(train_dir)
test_df = read_data(test_dir)
print("shape of train set before removing validation set = {}".format(train_df.shape))
print("shape of test set = {}".format(test_df.shape))

# step 3 : create validation split from train set by using day 66 for validation
val_df = train_df[train_df["f_1"] == 66]
train_df = train_df[train_df["f_1"] < 66]
print("shape of train set after removing validation set = {}".format(train_df.shape))
print("shape of validation set = {}".format(val_df.shape))

# step 4 : create masks for all three splits
train_df["train_mask"] = np.ones(len(train_df))
train_df["val_mask"] = np.zeros(len(train_df))
train_df["test_mask"] = np.zeros(len(train_df))

val_df["train_mask"] = np.zeros(len(val_df))
val_df["val_mask"] = np.ones(len(val_df))
val_df["test_mask"] = np.zeros(len(val_df))

test_df["train_mask"] = np.zeros(len(test_df))
test_df["val_mask"] = np.zeros(len(test_df))
test_df["test_mask"] = np.ones(len(test_df))

# step 5 : combine train and val split into one for further processing
full_df = pd.concat([train_df, val_df, test_df])

# step 6 : map each feature to it's type
date_feature_inds = np.arange(1)
date_features = np.char.add("f_", date_feature_inds.astype(str))

categorical_feature_inds = np.arange(2,42)
categorical_features = np.char.add("f_", categorical_feature_inds.astype(str))

numerical_feature_inds = np.arange(42,80)
numerical_features = np.char.add("f_", numerical_feature_inds.astype(str))

all_feature_inds =np.arange(1,80)
all_features = np.char.add("f_", all_feature_inds.astype(str))

labels = ['is_clicked', 'is_installed']

# step 7 : handle na values, normalize if necessary
full_df = full_df.replace('', np.nan)

for col in categorical_features:
    if (full_df[col].isnull().values.any()):
        full_df[col] = full_df[col].fillna(-1)

for col in labels:
    if (full_df[col].isnull().values.any()):
        full_df[col] = full_df[col].fillna(-1)

for col in numerical_features:
    if (full_df[col].isnull().values.any()):
        full_df[col] = full_df[col].replace(np.NaN, full_df[col].mean()) 

if normalize:
    print("WARNING: Using normalization of numerical features")
    mean =  full_df[numerical_features].mean()
    std = full_df[numerical_features].std()
    full_df[numerical_features] = (full_df[numerical_features] - mean) / std
print("data cleaning complete")

# step 8 : define roles for graph construction
labels = ['is_clicked', 'is_installed']
role1.sort()
role2.sort()
role3 = list(set(full_df.columns) - set(role1) - set(role2) - set(labels) )
role3.sort()

role_1_df = full_df.drop_duplicates(subset=role1).reset_index()[role1]
role_2_df = full_df.drop_duplicates(subset=role2).reset_index()[role2]

# step 9 : group using roles 
def get_group_idx(df, group_id, f=[]):
    k = [i for i in df.columns if i not in f][0]
    ret = df.groupby(by = f, as_index = False, sort=False)['f_0'].count().drop('f_0', axis=1)
    ret[f'group_{group_id}'] = ret.index
    ret2 = df.merge(ret, on=f)
    return ret2

full_graph_df = get_group_idx(full_df, 1, role1)
full_graph_df = get_group_idx(full_graph_df, 2, role2)
full_graph_df.rename(columns={"group_1": "src_id", "group_2": "dst_id"}, inplace=True)

# step 10 : write processed graph dataset to file
chunks = np.array_split(full_graph_df.index, 100) # split into 100 chunks for easy progress tracking
desired_cols = ['src_id', 'dst_id'] + role1 + role2 + role3 + labels

for chunk, subset in enumerate(tqdm(chunks, desc='Writing csv file')):
    chunk_df = full_graph_df[desired_cols].loc[subset]
    if chunk == 0: # first row
        chunk_df.to_csv(output_file, mode='w', compression='gzip', index=True)
    else:
        chunk_df.to_csv(output_file, header=None, mode='a', compression='gzip', index=True)
