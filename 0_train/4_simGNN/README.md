# Similarity GNN to generate "user-ad impression" embeddings
## Introduction
We model the ShareChat Dataset as a graph were each user-advertisement impression (each row in the provided Dataset) is a node and edges connect each node with the top 100 most similar impressions. 

The task is to predict if such impression will result into an app installation. To achieve that we trained a 3-layer GraphSAGE model over the similarity graph for supervised node classification task. 


### 1. Prepare the environment:
```
./build_env.sh
conda activate rec23_env
```

### 2. Calculate similary:
Two methods are used to calculate similarity between user-adv impressions.We first calculate similarity based on a subset of numerical features using faiss-gpu library. Secondly, we calculate similarity based on a subset of categoricals using sklearn NearestNeighbors and hamming distance.
```
python find_similarity_train.py
```

### 3. Build DGL Graph
We follow DGL CSVDataset format to ingest tabular data into a DGL graph: https://docs.dgl.ai/en/1.0.x/guide/data-loadcsv.html#guide-data-pipeline-loadcsv\

We formulate this GNN problem as in inductive setting, where a train graph is used to train the GNN and at inference time we use a different graph that also includes test nodes. 
During training we create a `trainval_graph` that contains all impressions up to day 66 included (train set)

Each graph is stored in its own directory and consists of 3 files : nodes.csv, edges.csv and meta.yaml needed to ingest graph into DGL

```
python build_graph_train.py
```

### 4. GNN Training and mapping of calculated embeddings to original data.
We trained a 3-layer GraphSAGE model over the similarity graph for supervised node classification task. Node classification in this case meaning predicting the "is_installed" label for each node in the graph. 

Once the GNN is trained we need to perform full graph inference (without sampling) over all the nodes and the generated node representation will be used as new features to improve the accuracy of an LGBM classifier. 

Because of the stochastic nature of GNNs we run the GNN training 10 times and select the experiments that provides the lowest validation logloss and best improvement on leaderboard. In this stage we use day 66 as the validation split
Note that the script below will call `node_classification_inductive_train.py` in a loop and save the pytorch model weights and logs for each of the 10 experiments.

```
./run10_gnn_exp.sh
```

### 5. Train LGBM
The generated embeddings from the full graph are combined with the original features, feature engineering (1_LearningFE), and bipartite-GNN embeddings (4_BiGNN) from self-supervised approach and fed into LGBM for training.
Please note that the path to the best simGNN embeddings need to be updated in the script below as indicated in the code
```
python LGBM_simGNN_train.py
```

