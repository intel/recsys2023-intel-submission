# Copyright (C) 2023 Intel Corporation

# SPDX-License-Identifier: MIT

# Downloaded from https://github.com/intel/graph-neural-networks-and-analytics

import pandas as pd
import numpy as np
import torch
import time
import yaml
import os
import argparse
from collections import OrderedDict


def main(args):
    IN_DATA = args.processed_data_path
    NODE_EMB = args.node_emb_path
    OUT_DATA_TRAIN = args.out_data_train_path
    OUT_DATA_TEST = args.out_data_test_path

    with open(args.tab2graph_cfg, "r") as file:
        config = yaml.safe_load(file)

    # 1.   load CSV file output of Classical ML edge featurization workflow
    print("loading processed data")
    start = time.time()
    df = pd.read_csv(IN_DATA, compression="gzip", engine="pyarrow")
    t_load_data = time.time()
    print("time lo load processed data", t_load_data - start)

    start = time.time()
    # 2.   Renumbering - generating node/edge ids starting from zero
    def column_index(series, offset=0):
        return {k: v + offset for v, k in enumerate(series.value_counts().index.values)}

    # create dictionary of dictionary to stare node mapping for all node types
    offset = 0
    dict = OrderedDict()
    # create mapping dictionary between original IDs and incremental IDs starting at zero
    # Note: because GNN is converting the graph to homogeneous we need the homogeneous mapping here
    # i,e: node_0: [0, x] node_1: [x,y] node_2: [y,z]
    col_map = {}
    for i, node in enumerate(config["node_columns"]):
        key = str(node + "_2idx")
        dict[key] = column_index(df[config["node_columns"][i]], offset=offset)
        new_col_name = node + "_Idx"
        col_map[node] = new_col_name
        # add new Idx to dataframe
        df[new_col_name] = df[config["node_columns"][i]].map(dict[key])
        offset = len(
            dict[key]
        )  # homogeneous mapping because that is how the embeddings will be returned bby GNN
    t_renum = time.time()
    print("re-enumerated column map (homogeneous mapping): ", col_map)
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
        emb = pd.DataFrame(df[col_map[node]].map(node_emb_dict).tolist()).add_prefix(
            "n" + str(i) + "_e"
        )
        df = df.join([emb])
        df.drop(
            columns=[col_map[node]],
            axis=1,
            inplace=True,
        )

    print("CSV output shape: ", df.shape)
    train_df = df[df["f_1"] < 67]
    test_df = df[df["f_1"] == 67]
    print(train_df.shape, test_df.shape)


    # write output combining the original columns with the new node embeddings as columns
    train_df.to_parquet(OUT_DATA_TRAIN, engine="pyarrow", index=False)
    test_df.to_parquet(OUT_DATA_TEST, engine="pyarrow", index=False)
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
        "--processed_data_path", 
        default="/localdisk/akakne/recsys2023/data/supGNN_graph_data/full_graph/full_edges.csv.gz",
        type=file, help="The path to the processed_data.csv"
    )
    parser.add_argument(
        "--node_emb_path",
        type=str,
        default="/localdisk/akakne/recsys2023/data/supGNN_graph_data/full_graph/full_node_emb.pt",
        help="The path to the node embedding file",
    )
    parser.add_argument(
        "--out_data_train_path",
        default="/localdisk/akakne/recsys2023/data/supGNN_graph_data/train_graph/train_edge_list_with_node_emb.parquet",
        type=str,
        help="The path to the csv data file with mapped node embeddings",
    )
    parser.add_argument(
        "--out_data_test_path",
        default="/localdisk/akakne/recsys2023/data/supGNN_graph_data/test_graph/test_edge_list_with_node_emb.parquet",
        type=str,
        help="The path to the csv data file with mapped node embeddings",
    )
    parser.add_argument(
        "--tab2graph_cfg",
        default="/localdisk/akakne/recsys2023/0_train/2_supGNN/recsys2graph.yaml",
        help="The path to the tabular2graph.yaml",
    )
    args = parser.parse_args()

    print(args)
    main(args)