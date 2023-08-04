# Instructions for running LearningFE Training for Recsys2023

## 1. Introduction

We prepare train data with variety of feature engineering to catogorical features and numerical features, which includes GlobalCountEncoding, NewValueOneHotEncoding and Indexing.

<div align="center">
  <img src="../../docs/graphs/fe_overview.png" width = "80%"/>
  <br>
  <center>Feature engineer pipeline overview. Role1 and Role2 refer to users and ads in this dataset. The same holds for rest 'Role' terms in this solution.</center>
</div>

After done feature engineeirng, processed train data will be used in two manners. 
* Training with LGBM to do inference on test data
* Combining with GNN embeddings and training with LGBM/XGBoost to do inference on test data. (How to generate GNN embeddings is decribed in [0_train/2_supGNN](/0_train/2_supGNN/), [0_train/3_BiGNN](/0_train/3_BiGNN) and [0_train/4_simGNN](/0_train/4_simGNN))

## 2. Getting start - Train data

```
# 0. We have assumption that train data has been located under data/sharechat_recsys2023_data/train

step1:
    open 0_data_prepare.ipynb and run through whole script

step2:
    open 1_train_LGBM.ipynb and run through whole script

```

## 3. Go to other folders under 0_train to create GNN embeddings

* [0_train/2_supGNN](/0_train/2_supGNN/)
* [0_train/3_BiGNN](/0_train/3_BiGNN)
* [0_train/4_simGNN](/0_train/4_simGNN)

## 4. Go for inference

please switch to [1_inference/1_LearningFE](/1_inference/1_LearningFE/)