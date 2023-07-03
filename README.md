codes for recsys2023 challenge

# Repo layout
``` bash
.
├── README.md
├── LICENSE
├── 0_train
├── 1_inference
├── utils.py
└── data
```

# instruction for running

* step1: prepare your data
  * put sharechat raw data under data/sharechat_recsys2023_data

* step2: Follow 0_train to complete model training and saving
  * use 1\_LearningFE to create processed data + encoder and LGBM model
  * use 2\_supGNN to create GNN model and xgboost model
  * use 3\_BiGNN to create GNN model
  * use 4\_simGNN to create GNN model and LGBM model

* step3: Follow 1_inference to do test data inference
  * use 1\_LearningFE to inference test data
  * use 2\_supGNN to inference test data
  * use 3\_BiGNN to create inference embeddings
  * use 4\_simGNN+BiGNN to inference test data
  * use 5\_ensemble.ipynb to get final result