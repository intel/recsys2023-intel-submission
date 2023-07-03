# LGBM using original features, FE, simGNN embeddings and embeddings from self-supervised bipartite graph

## 1. Prepare the environment:
We can reuse same environment as in the 4_simGNN training stage. 
```
conda activate rec23_env
```
### 2. Calculate similary for test nodes
Two methods are used to calculate similarity between user-adv impressions. We first calculate similarity based on a subset of numerical features using faiss-gpu library. Secondly, we calculate similarity based on a subset of categoricals using sklearn NearestNeighbors and hamming distance.
```
python find_similarity.py
```

### 3. Build DGL Graph 
We follow DGL CSVDataset format to ingest tabular data into a DGL graph: https://docs.dgl.ai/en/1.0.x/guide/data-loadcsv.html#guide-data-pipeline-loadcsv\

We formulate this GNN problem as in inductive setting, where a train graph is used to train the GNN and at inference time we use a different graph that also includes test nodes. 
During training we create a `full_graph` that contains all impressions including day 67

The graph directory consists of 3 files : nodes.csv, edges.csv and meta.yaml needed to ingest graph into DGL

```
python build_graph.py
```

### 4. GNN Inference

Due to the challenge constraint that test sessions should be predicted independently we had to exclude all similarities between test nodes during the graph modeling stage (Step 3) and implement a new method for 2-step layer-wise inference to avoid information leakage between test samples from the graph. Inference is firstly done on 'train nodes' and node representations at each layer are stored. Secondly, for every batch iteration of the test set the neighboring train nodes are reset to the saved per-layer node representation avoiding any inter-test set leakage.

In the script below please set the `ExpFolder` and `best_run_index` based on the best GNN training results. The bash script calls `node_classification_inductive_infer.py` to run GNN inference on full graph based on the pretrained model and `map_emb_single_recsys_parquet.py` to map the resulting embeddings to the original data prior to LGBM classification. 

```
./run_gnn_infer.sh
```
## 5. Run pre-trained LGBM prediction on test set 
Looad pretrained model and run inference on test set. This requires step 3_BiGNN train and inference are completed since those embeddings are used as input in LGBM
```
python LGBM_simGNN_submission.py
```


