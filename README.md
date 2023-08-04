# Intel solution for RecSys challenge 2023

This repository provides the official implementation of Intel solution for RecSys 2023 challenge from the paper: **Graph Enhanced Feature Engineering for Privacy Preserving Recommendation Systems**.

The solution for RecSys Challenge 2023 leverages a novel feature classification method to categorize anonymous features into different groups and apply enhanced feature engineering and graph neural networks to reveal underlying information and improve prediction accuracy. This solution can also be generalized to other privacy preserving recommendation systems. Our team name is LearningFE, final submission got score as 5.892977 and ranks at 2’nd on the leaderboard. 

# Introduction
## RecSys2023 Challenge
[RecSys 2023 challenge](http://www.recsyschallenge.com/2023/) focus on online advertising, improving deep funnel optimization, and user privacy. The dataset corresponds to impressions of users and ads from ShareChat + Moj app, where each impression (a set of 80 features) is an advertisement (ad) that was shown to a user and whether that resulted in a click or an install of the application corresponding to the advertisement. The problem is to predict the probability of an impression resulting in application installation.

## Intel's solution
The core idea of our solution is to augment and enrich the original dataset using (i) privacy preserving feature engineering, (ii) bipartite graph neural networks, and (iii) similarity-based graph neural network. The augmented
datasets from the above approaches are used with gradient-boosted decision trees (XGBoost and LGBM) to predict the probability of installation. Finally, we ensemble the three solutions together to obtain our final result.

# Architechture
Our solution is the ensemble of 3 models by using different training methods and feature sets as showed in below graph. We generate new features from several mehtods: (i) privacy preserving feature engineering output, (ii) supervised bipartite GNN (BiGNN) embeddings, (iii) self-supervised BiGNN embeddings and (iv) similarity Graph GNN (simGNN) embeddings. The first model is a LightGBM model trained with enhanced feature sets. The second model  is an XGBoost model trained with enhanced feature sets and supervised BiGNN embeddings. The third model is a LightGBM model trained with enhanced feature sets, simGNN embeddings and self-supervised BiGNN embeddings.

<div align="center">
  <img src="docs/graphs/ensemble.png" width = "90%"/>
  <br>
  <center>Model architechture overview.</center>
</div>

## Key Components
### Feature engineer pipeline 
We propose a novel [feature engineering pipeline](0_train/1_LearningFE) for privacy-preservation dataset, which is capable of enriching the features' expressiveness based on feature distribution characteristics. This method comprises of three major steps: a) analysis and classification, b) feature engineering, and c) feature selection. 

<div align="center">
  <img src="docs/graphs/fe_overview.png" width = "80%"/>
  <br>
  <center>Feature engineer Pipeline Overview.</center>
</div>

### GNN embedding feature
We employ GNN to generate the embedding feature as input of the GBDT model for installation prediction. Depends on the graph representation, we produce three type of GNN embedding feature: a) [self-supervised GNN](0_train/2_supGNN), b) [supervised GNN](0_train/3_BiGNN), c) [Similarity GNN](4_simGNN).

# How to run
* step1: prepare your data
  * put sharechat raw data under data/sharechat_recsys2023_data

* step2: Follow 0_train to complete model training and saving
  * use 1\_LearningFE to create processed data + encoder and LGBM model, see [here](0_train/1_LearningFE/README.md) for details
  * use 2\_supGNN to create GNN model and xgboost model, see [here](0_train/2_supGNN/README.md) for details
  * use 3\_BiGNN to create GNN model, see [here](0_train/3_BiGNN/README.md) for details
  * use 4\_simGNN to create GNN model and LGBM model, see [here](0_train/4_simGNN/README.md) for details

* step3: Follow 1_inference to do test data inference
  * use 1\_LearningFE to inference test data, see [here](1_inference/1_LearningFE/README.md) for details
  * use 2\_supGNN to inference test data, see [here](1_inference/1_LearningFE/README.md) for details
  * use 3\_BiGNN to create inference embeddings, see [here](1_inference/2_supGNN/README.md) for details
  * use 4\_simGNN+BiGNN to inference test data, see [here](1_inference/3_BiGNN/README.md) for details
  * use 5\_ensemble.ipynb to get final result, see [here](1_inference/4_ensemble.ipynb) for details

# Citation
If you use this codebase, or otherwise found our work valuable, please cite:

```
@inproceedings{intelsolutionforrecsys2023,
  title={Graph Enhanced Feature Engineering for Privacy Preserving Recommendation System},
  author={CHENDI XUE, XINYAO WANG, Yu Zhou, Poovaiah Palangappa, Ravi Motwani, Rita Brugarolas Brufau, Aasavari Dhananjay Kakne, Ke Ding, Jian Zhang},
  booktitle={RecSys 2023},
  year={2023}
}
```