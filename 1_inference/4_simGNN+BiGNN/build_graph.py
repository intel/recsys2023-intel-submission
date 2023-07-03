import pandas as pd
import numpy as np
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 150)

import os
from tqdm import tqdm
import yaml
import dgl

## To generate graphs from CSV files we follow DGL CSVDataset format specs: https://docs.dgl.ai/en/1.0.x/guide/data-loadcsv.html#guide-data-pipeline-loadcsv
## this script generates the metal.yml, nodes.csv and edges.csv file

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

def fix_na(df):
    df = df.fillna(0)
    for col in df.select_dtypes([float]):
        v = col
        if np.array_equal(df[v], df[v].astype(int)):
            df[v] = df[v].astype(int, copy = False)
    return df
train_df = fix_na(train_df)
test_df = fix_na(test_df)

#Input Path to similarity distances and indices baseed on numericals and categoricals.
sim_data_path_num="../../data/4_simGNN/similarity"
ind_file_num="full_day45_67_sim100_indices_19num.npy"
sim_data_path_cat=sim_data_path_num
ind_file_cat = "full_day45_67_sim100_indices_cat.npy"

##output graphs
output_graph="../../data/4_simGNN/release_graph_full"
if not os.path.exists(output_graph):
    os.makedirs(output_graph)

print("concat train and test")
train_test_df = pd.concat([train_df,test_df], ignore_index=True)
train_test_df_ori = train_test_df.copy()
train_test_df.to_csv("./recsys_data45_67.csv", index=False)

# feature_id_list=[42,43,44,45,46,47,48,49,50, 51,52,53,54, 55, 56, 57,58, 59,60,61, 62, 63,64,65, 66, 67, 68, 69, 70,71,72,73,74,75,76,77,78,79]
# train_columns = [f"f_{i}" for i in feature_id_list]

# #min-max normalizing with train_df (not to leak any info from test_df). 
# # There is only f_49 where day 67 has a larger value
# # x=train_test_df.groupby(['f_1'])[train_columns].aggregate(['min', 'max', 'mean']) 
# train_test_df[train_columns] = (train_test_df[train_columns] - train_df[train_columns].min()) / (train_df[train_columns].max()-train_df[train_columns].min())

##STEP 3: Load similarity edges based on similarity indices (from numerical)
print("loading numerical sim links")
indices = np.load(os.path.join(sim_data_path_num,ind_file_num))
def transform_indices_list(indices):
    indices_df = pd.DataFrame(indices)
    indices_df['list'] = indices_df.apply(lambda x: x.tolist(), axis=1)
    indices_df_list = indices_df['list']
    del indices_df
    print(indices_df_list)
    return indices_df_list

indices_df_list= transform_indices_list(indices)
edge_df=pd.DataFrame()
edge_df['indices_list']=indices_df_list
edge_df = edge_df.explode('indices_list')
edge_df.reset_index(inplace=True)
edge_df.rename(columns={"index": "src_id", "indices_list": "dst_id"},inplace=True)

##STEP 3: Load similarity edges based on similarity indices (from categorical)
print("loading categorical sim links")
indices_cat = np.load(os.path.join(sim_data_path_cat,ind_file_cat))
print(indices_cat.shape)

indices_df_list_cat= transform_indices_list(indices_cat)
edge_df_cat=pd.DataFrame()
edge_df_cat['indices_list']=indices_df_list_cat
edge_df_cat = edge_df_cat.explode('indices_list')
edge_df_cat.reset_index(inplace=True)
edge_df_cat.rename(columns={"index": "src_id", "indices_list": "dst_id"},inplace=True)

##STEP 4: Combine all similarity edges (numerical and categorical)
print("combining numerical links with categorical based links")
combined_edges=pd.concat([edge_df,edge_df_cat])
combined_edges.shape

##STEP 6: Save edges for full graph --> includes test nodes but no test-test edges
print("save full graph edges_0.csv w numerical and categorical edges excluding test to test edges")
test_ids=train_test_df.index.max()-test_df.shape[0]
edge_df_noTestTest=combined_edges[(combined_edges.src_id < test_ids) | (combined_edges.dst_id < test_ids)]
edge_df_noTestTest.to_csv(os.path.join(output_graph,"edges_0.csv"), index=False, header=["src_id","dst_id"])
print("shape of full graph without test-test", edge_df_noTestTest.shape)


# STEP 7: prepare node file full graph
train_test_df['train_mask']= 1
train_test_df.loc[train_test_df['f_1'] == 66, 'train_mask'] = 0
train_test_df.loc[train_test_df['f_1'] == 67, 'train_mask'] = 0
train_test_df['val_mask']=0
train_test_df.loc[train_test_df['f_1'] == 66, 'val_mask'] = 1
train_test_df['test_mask']=0
train_test_df.loc[train_test_df['f_1'] == 67, 'test_mask'] = 1
print(train_test_df.columns)
print(train_test_df['train_mask'].sum(), train_test_df['val_mask'].sum(),train_test_df['test_mask'].sum())


feat_keys=['f_1', 'f_2', 'f_3', 'f_4', 'f_5', 'f_6', 'f_8', 'f_9', 'f_10', 'f_11', 'f_12', 'f_13', 'f_14', 'f_15', 'f_16', 'f_17', 'f_18',
       'f_19', 'f_20', 'f_21', 'f_22', 'f_23', 'f_24', 'f_25', 'f_26', 'f_27', 'f_28', 'f_29', 'f_30', 'f_31', 'f_32', 'f_33', 'f_34', 'f_35',
       'f_36', 'f_37', 'f_38', 'f_39', 'f_40', 'f_41', 'f_42', 'f_43', 'f_44', 'f_45', 'f_46', 'f_47', 'f_48', 'f_49', 'f_50', 'f_51', 'f_52',
       'f_53', 'f_54', 'f_55', 'f_56', 'f_57', 'f_58', 'f_59', 'f_60', 'f_61', 'f_62', 'f_63', 'f_64', 'f_65', 'f_66', 'f_67', 'f_68', 'f_69',
       'f_70', 'f_71', 'f_72', 'f_73', 'f_74', 'f_75', 'f_76', 'f_77', 'f_78', 'f_79']
train_test_df["feat_as_str"] = train_test_df_ori[feat_keys].astype(str).apply(",".join, axis=1)

print("save full graph nodes_0.csv")
train_test_df[['is_clicked', 'is_installed', 'train_mask', 'val_mask','test_mask','f_0','feat_as_str']].to_csv(os.path.join(output_graph,"nodes_0.csv"), index=True, header=['is_clicked', 'is_installed','train_mask', 'val_mask','test_mask','f_0','feat'],index_label="node_id")


# STEP 9: save meta.yaml required for DGL graph ingestion
print("create meta yamls")
# write meta.yaml for full graph
meta_yaml = """
dataset_name: recsys2023_days45_67
edge_data:
- file_name: edges_0.csv
  etype: [impIdx, sim, impIdx]
node_data:
- file_name: nodes_0.csv
  ntype: impIdx
"""

meta = yaml.safe_load(meta_yaml)

with open(os.path.join(output_graph, "meta.yaml"), "w") as file:
  yaml.dump(meta, file)

##To test its properly formated
print("ingest graph into DGL")
dataset = dgl.data.CSVDataset(output_graph, force_reload=False)
g=dataset[0]
print(g)
