import argparse
import pandas as pd
import sys
import xgboost as xgb

def train_xgb(data_df, label, params, trees, eval_every, model_path):
    DM_train = xgb.DMatrix(data=data_df.drop(columns=[label, "f_0", "f_1"]), label=data_df[label])
    xgb_model = xgb.train(params=params,
            dtrain=DM_train,
            evals=[(DM_train,'train')],
            num_boost_round=trees,
            early_stopping_rounds=trees,
            verbose_eval=eval_every)
    xgb_model.save_model(model_path)

def read_data(path):
    df = pd.read_parquet(path, engine="pyarrow")
    selected_features = ['dow']
    selected_features += [f"f_{i}" for i in range(0,80)]
    selected_features += [f"f_{i}_CE" for i in [2,4,6,13,15,18] + [78,75,50,20,24]]
    selected_features += [f"f_{i}_idx" for i in range(2,23) if i not in [2,4,6,15]]
    selected_features += [col for col in df.columns if 'n0' in col or "n1" in col]
    selected_features += ["is_installed"]
    ignore_cols = [col for col in df.columns if col not in selected_features]
    df.drop(columns=ignore_cols, inplace=True)
    print("GNN-boosted features have shape = {}".format(df.shape))
    return df


def main(arglist):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        default="/localdisk/akakne/recsys2023/data/supGNN_graph_data/train_graph/merged_train_data_FE_and_supgnn.parquet",
        help="provide path to GNN-boosted data",
    )
    parser.add_argument(
        "--model_path",
        default="/localdisk/akakne/recsys2023/data/supGNN_graph_data/train_graph/xgb_model.pt",
        help="provide path to save predictions using final model",
    )
    args = parser.parse_args(arglist)

    # step 1 : reading original features and GNN-boosted features into dataframes
    df = read_data(args.data_path)

    # step 2 : define optimal parameters for baseline as well as final model 
    params = {'max_depth':6,
        'learning_rate':0.01,
        'subsample':0.8,
        'colsample_bytree':0.8,
        'eval_metric':['logloss'],
        'objective':'binary:logistic',
        'tree_method':'hist',
        "random_state":42}
    trees = 5000
    eval_every = 500
    label = "is_installed"

    # step 3 : train xgboost model, save predictions to file
    print("Training final model")
    train_xgb(df, label, params, trees, eval_every, args.model_path)

if __name__ == "__main__":
    main(sys.argv[1:])
