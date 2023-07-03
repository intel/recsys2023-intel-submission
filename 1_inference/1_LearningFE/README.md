# Instructions for running LearningFE inference for Recsys2023

## 1. Getting start - inference data

```
# 0. We have assumption that all intermidate encoding dictionary and trained model has been located under data/1_LearningFE/

step1:
    open 0_data_prepare.ipynb and run through whole script

step2:
    open 1_train_LGBM.ipynb and run through whole script

```

## 2. Go to other folders under 1_inference to get GNN embeddings for test data

* [1_inference/2_supGNN](/1_inference/2_supGNN/)
* [1_inference/3_BiGNN](/1_inference/3_BiGNN)
* [1_inference/4_simGNN+BiGNN](/1_inference/4_simGNN+BiGNN)

## 3. Ensemble with all the test inference

```
open 1_inference/4_ensemble.ipynb to combine all the test inference

```
