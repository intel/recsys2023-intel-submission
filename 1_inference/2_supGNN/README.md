# Supervised Graph Neural Network (aka supGNN) paired with downstream XGBoost model 

## Introduction 
Please refer to 0_train/2_supGNN/README.md to complete training
Here is remain steps for inference 

### 1. Convert tabular data to edge list for inference
Using both `./sharechat_recsys2023_data/train` and  `./sharechat_recsys2023_data/test` folders, we create `./supGNN_graph_data/full_edges.csv.gz` which will be used for supGNN inference later. 
```bash
python3 1_inference/2_supGNN/convert_full_tabular_data_to_edge_list.py
```
### 2. Create CSVDataset from edge list
Convert full graph into full CSVDataset (for GNN inference)
```bash
python3 1_inference/2_supGNN/convert_full_edge_list_to_CSVDataset.py
```
Please see newly created files inside `./data/supGNN_graph_data/full_graph/recsys_graph` folder. 

### 3. Run supervised Graph Neural Network to generate node embeddings
Run inference on full graph using saved GNN model </br>
```bash
python3 1_inference/2_supGNN/infer_supervised_graphsage.py
```

### 4. Map GNN embeddings to original features 
```bash
python3 1_inference/2_supGNN/map_node_emb_to_edge_list.py
```
this script will map GNN-generated node embeddings to their respective edges and save GNN-boosted features for train and test split separately. 

### 5. Merge supGNN data with Feature Engineered data 
Merge test supGNN data with test FE data 
```bash
python3 1_inference/2_supGNN/merge_test_FE_and_test_supgnn_data.py
```

### 6. Run XGBoost Inference 
Infer XGB on merged test data
```bash
python3 1_inference/2_supGNN/infer_xgb.py
```

Finally, we get `./data/supGNN_graph_data/test_graph/submission.csv` as our final submission file which we upload to the leaderboard for evaluation on test set. 