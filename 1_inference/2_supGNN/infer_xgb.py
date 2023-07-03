import argparse
import pandas as pd
import sys
import xgboost as xgb

def infer_xgb(test_df, saved_model_path, pred_file_name):
    DM_test = xgb.DMatrix(data=test_df.drop(columns=["f_0", "f_1"]))
    xgb_model = xgb.Booster()
    xgb_model.load_model(saved_model_path)
    scores = xgb_model.predict(DM_test)
    sub_df = pd.DataFrame({"RowId" : test_df["f_0"], "is_installed" : scores, "is_clicked" : scores})
    sub_df.to_csv(pred_file_name, index=False, sep="\t")

def read_data(path):
    df = pd.read_parquet(path, engine="pyarrow")
    selected_features = ['dow']
    selected_features += [f"f_{i}" for i in range(0,80)]
    selected_features += [f"f_{i}_CE" for i in [2,4,6,13,15,18] + [78,75,50,20,24]]
    selected_features += [f"f_{i}_Count" for i in range(2,23) if i not in [2,4,6,15]]
    selected_features += [col for col in df.columns if 'n0' in col or "n1" in col]
    ignore_cols = [col for col in df.columns if col not in selected_features]
    df.drop(columns=ignore_cols, inplace=True)
    print("GNN-boosted features have shape = {}".format(df.shape))
    return df


def main(arglist):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        default="/localdisk/akakne/recsys2023/data/supGNN_graph_data/test_graph/merged_test_data_FE_and_supgnn.parquet",
        help="provide path to GNN-boosted data",
    )
    parser.add_argument(
        "--model_path",
        default="/localdisk/akakne/recsys2023/data/supGNN_graph_data/train_graph/xgb_model.pt",
        help="provide path to save predictions using final model",
    )
    parser.add_argument(
        "--submission_path",
        default="/localdisk/akakne/recsys2023/data/supGNN_graph_data/test_graph/submission.csv",
        help="provide path to save predictions using final model",
    )

    args = parser.parse_args(arglist)

    # step 1 : reading original features and GNN-boosted features into dataframes
    df = read_data(args.data_path)

    print("Running inference on test split using saved xgb model")
    infer_xgb(df, args.model_path, args.submission_path)


if __name__ == "__main__":
    main(sys.argv[1:])
