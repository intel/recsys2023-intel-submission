# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT
# Downloaded from https://github.com/intel/graph-neural-networks-and-analytics   
import pandas as pd
import numpy as np
import csv
import torch
import time
import yaml
import os
import argparse
from collections import OrderedDict

def main(args):
    IN_DATA = args.processed_data_path
    NODE_EMB = args.model_emb_path + "/" + args.node_emb_name + ".pt"
    OUT_DATA = args.out_data_path

    with open(args.tab2graph_cfg,'r') as file:
        config = yaml.safe_load(file)

    # 1.   load CSV file output of Classical ML edge featurization workflow
    print("loading processed data")
    start = time.time()
    df = pd.read_csv(IN_DATA, engine='pyarrow')
    t_load_data = time.time()
    print("time lo load processed data", t_load_data - start)

    start = time.time()
    # 2.   Renumbering - generating node/edge ids starting from zero
    def column_index(series, offset=0):
        return {k: v + offset for v, k in enumerate(series.value_counts().index.values)}

    #create dictionary of dictionary to stare node mapping for all node types
    offset=0
    dict= OrderedDict()
    #create mapping dictionary between original IDs and incremental IDs starting at zero
    col_map={}
    for i, node in enumerate(config['node_columns']):
        a1=np.sort( df[config['node_columns'][i]].unique() )
        a2=np.arange(0, len(df[config['node_columns'][i]].unique()) )
        if ( np.array_equal(a1, a2) ):
            print("Node ID's already numbered")
            col_map[node]=node
        else:
            key=str(node+"_2idx")
            dict[key] = column_index(df[config['node_columns'][i]], offset=offset)
            new_col_name = node + '_Idx'
            col_map[node]=new_col_name
            #add new Idx to dataframe
            df[new_col_name] = df[config['node_columns'][i]].map(dict[key])
            #offset = len(dict[key]) #remove if doing hetero mappint where all types start from zero
    t_renum = time.time()
    print("re-enumerated column map: ", col_map)
    print("time to renumerate", t_renum - t_load_data)
    

    # 6.   load node embeddings from file, add them to edge features and save file for Classic ML workflow (since model is trained as homo, no mapping needed.)
    print("Loading embeddings from file and adding to preprocessed CSV file")
    if not os.path.isfile(NODE_EMB):
        raise argparse.ArgumentTypeError(
            '"{}" is not an existing file'.format(NODE_EMB)
        )
    node_emb = torch.load(NODE_EMB)
    node_emb_arr = node_emb.cpu().detach().numpy()
    node_emb_dict = {i: val for i, val in enumerate(node_emb_arr)}

    for i, node in enumerate(col_map.keys()):
        emb = pd.DataFrame(df[col_map[node]].map(node_emb_dict).tolist()).add_prefix("n"+str(i)+"_e")
        df = df.join([emb])
        df.drop(columns=[col_map[node]], axis=1,inplace=True,)
    
    print("CSV output shape: ", df.shape)

    # write output combining the original columns with the new node embeddings as columns
    df.to_csv(OUT_DATA+".gz", compression='gzip', index=False)
    print("Time to append node embeddings to edge features CSV", time.time() - start)

def file(raw_path):
    if not os.path.isfile(raw_path):
        raise argparse.ArgumentTypeError(
            '"{}" is not an existing file'.format(raw_path)
        )
    return os.path.abspath(raw_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MapEmb and save")
    parser.add_argument(
        "--processed_data_path", type=file, help="The path to the processed_data.csv"
    )
    parser.add_argument(
        "--model_emb_path",
        help="The path to the pt files generated in training",
    )
    parser.add_argument(
        "--node_emb_name",
        type=str,
        default="node_emb",
        help="The path to the node embedding file",
    )
    parser.add_argument(
        "--out_data_path",
        type=str,
        help="The path to the csv data file with mapped node embeddings",
    )
    parser.add_argument(
        "--tab2graph_cfg", required=True, help="The path to the tabular2graph.yaml",
    )
    args = parser.parse_args()

    print(args)
    main(args)