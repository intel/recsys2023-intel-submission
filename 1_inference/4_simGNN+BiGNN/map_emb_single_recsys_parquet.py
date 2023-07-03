import pandas as pd
import numpy as np
import csv
import torch
import time
#import graphsage_fraud_transductive
import yaml
import os
import argparse
from collections import OrderedDict


def main(args):
    IN_DATA = args.processed_data_path
    NODE_EMB = args.emb_path 
    OUT_DATA = args.out_data_path


    # 1.   load CSV file output of GNN inference
    print("loading original data")
    start = time.time()
    df = pd.read_csv(IN_DATA)
    t_load_data = time.time()
    print("time lo load processed data", t_load_data - start)

   
    # 2.   load node embeddings from file
    print("Loading embeddings from file and adding to preprocessed CSV file")
    if not os.path.isfile(NODE_EMB):
        raise argparse.ArgumentTypeError(
            '"{}" is not an existing file'.format(NODE_EMB)
        )
    node_emb = torch.load(NODE_EMB)
    node_emb_arr = node_emb.cpu().detach().numpy()
    node_emb_df = pd.DataFrame(node_emb_arr)
    print(node_emb_df.shape)
    df=df.join(node_emb_df)

    print("CSV output shape: ", df.shape)

    # write output combining the original columns with the new node embeddings as columns
    df.columns = df.columns.astype(str)
    df.to_parquet(OUT_DATA, index=False, engine='pyarrow')
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
        "--processed_data_path", type=file, help="The path to the processed_data.csv",
        default="./recsys_data45_67.csv",
    )
    parser.add_argument(
        "--emb_path",
        default= "./node_emb_train_test.pt",
        help="The path to the pt files generated in training",
    )
    parser.add_argument(
        "--out_data_path",
        type=str,
        help="The path to the csv data file with mapped node embeddings",
    )
    args = parser.parse_args()

    print(args)
    main(args)
