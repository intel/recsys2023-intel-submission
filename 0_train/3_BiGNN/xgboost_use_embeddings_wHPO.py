#!/usr/bin/env python
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT
import pandas as pd
import numpy as np
import xgboost as xgb
import sklearn

target_label = 'is_installed'
def read_data_graph(file):
    print ('reading file ', file)
    df = pd.read_csv(file, compression='gzip', engine="pyarrow")
    return df

file = "./graph_data/tabular_with_gnn_emb.csv.gz"
emb_fullds_df = read_data_graph(file)


date_feature_inds = np.arange(1)
date_features = np.char.add("f_", date_feature_inds.astype(str))
categorical_feature_inds = np.arange(2,42)
categorical_features = np.char.add("f_", categorical_feature_inds.astype(str))
numerical_feature_inds = np.arange(42,80)
numerical_features = np.char.add("f_", numerical_feature_inds.astype(str))

all_feature_inds =np.arange(1,80)
all_features = np.char.add("f_", all_feature_inds.astype(str))
unimportant_inds = (79, 39, 32, 77, 71) #, 52, 41, 33, 61, 38, 54, 15, 53, 63, 17)
important_feature_inds = np.delete(all_feature_inds, np.array(unimportant_inds)-1 )
important_features = np.char.add("f_", important_feature_inds.astype(str))

sorted_emb_df = emb_fullds_df.sort_values(by = list(all_features), ascending=False)
df = sorted_emb_df

val_df = df[df["f_1"]==66]
train_df = df[df["f_1"]<66]
test_df = df[df["f_1"]==67]

all_features_after_embedding =list(set(train_df.columns) -  set(["is_clicked", "is_installed", "train_mask", "val_mask", "test_mask", "f_0"]))
all_features_after_embedding.sort(reverse=False)
all_features.sort()

import sklearn
import optuna
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def objective(trial):

    DM_train = xgb.DMatrix(data=train_df[all_features], label=train_df[target_label])
    DM_val = xgb.DMatrix(data=val_df[all_features], label=val_df[target_label])
    param = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method":"hist",
        "random_state": 42,
        "lambda": trial.suggest_loguniform("lambda", 1e-8, 1.0),
        "alpha": trial.suggest_loguniform("alpha", 1e-8, 1.0),
        "learning_rate":trial.suggest_uniform("learning_rate", 0.01, 0.5), 
        "subsample": trial.suggest_uniform("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.5, 1.0), 
        "scale_pos_weight": 1,
        "max_depth": trial.suggest_int("max_depth", 1, 9),
    }

    # Add a callback for pruning.
    pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "validation-logloss")
    bst = xgb.train(param, DM_train, evals=[(DM_val, "validation")], callbacks=[pruning_callback], num_boost_round=250, verbose_eval=25)
    preds = bst.predict(xgb.DMatrix(val_df[all_features]))
    # pred_labels = np.rint(preds)
    y = val_df[target_label].to_numpy()
    avg_logloss =  np.mean(sklearn.metrics.log_loss(y, preds))
    return avg_logloss
study = optuna.create_study(study_name="xgboost_baseline", direction='minimize', load_if_exists=True)
study.optimize(objective, n_trials=20)
print(study.best_trial)



print(study.best_params)

params = study.best_params

params.update( {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method":"hist",
    "random_state": 42,
    "scale_pos_weight": 1,
})
params['eval_metric'] =['auc','mae', 'logloss', 'aucpr']
DM_train = xgb.DMatrix(data=train_df[all_features], label=train_df[target_label])
DM_val = xgb.DMatrix(data=val_df[all_features], label=val_df[target_label])
xgb_model_install = xgb.train(params=params, 
          dtrain=DM_train,
          evals=[(DM_train,'train'),(DM_val,'val')],
          num_boost_round=1000,
          early_stopping_rounds=1000,
          verbose_eval=25)



def objective_gnn(trial):

    DM_train = xgb.DMatrix(data=train_df[all_features_after_embedding], label=train_df[target_label])
    DM_val = xgb.DMatrix(data=val_df[all_features_after_embedding], label=val_df[target_label])
    param = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method":"hist",
        "random_state": 42,
        "lambda": trial.suggest_loguniform("lambda", 1e-8, 1.0),
        "alpha": trial.suggest_loguniform("alpha", 1e-8, 1.0),
        "learning_rate":trial.suggest_uniform("learning_rate", 0.01, 0.5), 
        "subsample": trial.suggest_uniform("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.5, 1.0), 
        "scale_pos_weight": 1,
        "max_depth": trial.suggest_int("max_depth", 1, 9),
    }

    # Add a callback for pruning.
    pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "validation-logloss")
    bst = xgb.train(param, DM_train, evals=[(DM_val, "validation")], callbacks=[pruning_callback], num_boost_round=250, early_stopping_rounds=25, verbose_eval=25)
    preds = bst.predict(xgb.DMatrix(val_df[all_features_after_embedding]))
    y = val_df[target_label].to_numpy()
    avg_logloss =  np.mean(sklearn.metrics.log_loss(y, preds))
    return avg_logloss
study = optuna.create_study(study_name="xgboost_gnn", direction='minimize', load_if_exists=True)
study.optimize(objective_gnn, n_trials=20)
print(study.best_trial)



print(study.best_params)

params = study.best_params

params.update( {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method":"hist",
    "random_state": 42,
    "scale_pos_weight": 1,
})
params['eval_metric'] =['auc','mae', 'logloss', 'aucpr']
# DM_train = xgb.DMatrix(data=x_train_cat, label=y_train)

df = sorted_emb_df

val_df = df[df["f_1"]==66]
train_df = df[df["f_1"]<66]
test_df = df[df["f_1"]==67]

DM_train = xgb.DMatrix(data=train_df[all_features_after_embedding], label=train_df[target_label])
DM_val = xgb.DMatrix(data=val_df[all_features_after_embedding], label=val_df[target_label])
xgb_model_with_emb = xgb.train(params=params, 
          dtrain=DM_train,
          evals=[(DM_train,'train'),(DM_val,'val')],
          num_boost_round=1000,
          early_stopping_rounds=1000,
          verbose_eval=25)


