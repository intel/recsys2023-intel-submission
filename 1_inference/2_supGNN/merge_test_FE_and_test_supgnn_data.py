import argparse
import pandas as pd
import sys

def merge_datasets(dataset, embedding, feats_to_drop=None):
    """returns a merged dataframe using dataset and embedding dataframes on f_0 column"""
    copy_fullds_df = dataset
    copy_emb_df = embedding
    reset_ind_copy_fullds_df = copy_fullds_df.reset_index(drop=True)
    reset_ind_copy_emb_df = copy_emb_df.reset_index(drop=True).drop(columns=feats_to_drop)
    assert reset_ind_copy_fullds_df.shape[0] == reset_ind_copy_emb_df.shape[0]
    rearranged_df = reset_ind_copy_fullds_df.merge(reset_ind_copy_emb_df, on='f_0', how='left', suffixes=('_1', '_2'))
    assert rearranged_df.shape[0] == reset_ind_copy_fullds_df.shape[0]
    return rearranged_df

def read_data(file, cols_to_drop=[]):
    """takes in a .gz or .csv.gz or .parquet file, prints it's columns nicely and drops unnecessary columns"""
    ftype = file.split(".")[-1]
    if ftype == 'gz':
        df = pd.read_csv(file, compression='gzip', engine='pyarrow')
    elif ftype == 'csv':
        df = pd.read_csv(file, engine='pyarrow')
    elif ftype == 'parquet':
        df = pd.read_parquet(file, engine='pyarrow')
    else:
        print("File type not supported, please provide .csv or .csv.gz or .parquet file")
        return 
    if cols_to_drop:
        df.drop(columns=cols_to_drop, inplace=True)
    return df

def print_cols(df, name):
    print("-"*100)
    print("printing columns of {}".format(name))
    cols = list(df.columns)
    for i in range(0, len(cols), 5):
        print(cols[i : i + 5])
    print("-"*100)

def main(arglist):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fe_data_path",
        default="/localdisk/akakne/recsys2023/data/1_LearningFE/test_processed.parquet",
        help="provide path to FE data",
    )
    parser.add_argument(
        "--supgnn_data_path",
        default="/localdisk/akakne/recsys2023/data/supGNN_graph_data/test_graph/test_edge_list_with_node_emb.parquet",
        help="provide path to FE data",
    )
    parser.add_argument(
        "--merged_data_path",
        default="/localdisk/akakne/recsys2023/data/supGNN_graph_data/test_graph/merged_test_data_FE_and_supgnn.parquet",
        help="provide path to FE data",
    )
    args = parser.parse_args(arglist)

    # step 1 : read input into dataframes
    test_fe_df = read_data(args.fe_data_path)
    test_supgnn_df = read_data(args.supgnn_data_path, ["", "src_id", "dst_id", "train_mask", "val_mask", "test_mask"])
    print(test_fe_df.shape, test_supgnn_df.shape)

    # step 2 : merge edges and FE10 
    orig_features = ['f_{}'.format(i) for i in range(1, 80)]

    test_df = merge_datasets(test_fe_df, test_supgnn_df, orig_features)
    print("merged test df shape = {}".format(test_df.shape))

    # step 3 : write merged dataframe to file
    test_df.to_parquet(args.merged_data_path, engine="pyarrow", index=False)



if __name__ == "__main__":
    main(sys.argv[1:])