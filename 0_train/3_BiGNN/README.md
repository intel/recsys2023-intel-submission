# Instructions for running the self-supervised GNN for RecSys2023

## 1. Introduction:
### 1.1 Modeling as a graph
    This section of the code models the dataset into a graph of two types of nodes -- role-1 and role-2. Each impression of the dataset is modelled as an edge between a node of type role-1 and a node of type role-2. 

    Since we have no information as what features belong users and advertisements (we use the term 'role' to refer to either of them), we ran experiments to try out various feature-to-role assignments and measuring the performance on the validation dataset. Finally, we settled with the following assignment
        - Role 1: f_6
        - Role 2: f_2, f_4, and f_16

    It is important to note that these roles need not refer to users and advertisements individually. We only expect them to refer to groups of users and groups of advertisements. 

    Therefore, every impression connects a node of type role 1 and node of type role 2. 

<div align="center">
  <img src="../../docs/graphs/supGNN_pipeline.png" width = "80%" />
  <br>
  <center>Supervised bipartite GNN pipeline.</center>
</div>

### 1.1 Summary
    The graph is modeled with nodes corresponding to roles 1 and 2 and edges corresponding to every impression

## 2. Setup:
    Place the Sharechat Recsys2023 datasets in the folder $GIT_ROOT/dataset/sharechat_recsys2023_data
        a. $GIT_ROOT/data/sharechat_recsys2023_data/train/ --> place the training dataset here

## 3. Run instructions
### 3.1 Flow (no run commands in this section; only explanation)
Due to the inherent Random nature of the GNN edge sampling, we perform monte carlo simulation (10 runs in our case) and pick the embedding set that performs best on the validation (day 66) dataset. The following steps are performed in a loop

1. Group dataset into nodes based on features f_6 for role-1 and f_2, f_4, and f_16 for role-2
2. Create train and val masks for the training dataset: 
    - training data uses all days until and including day 65
    - validation data uses day 66
2. Create a yaml file to communicate the graph information to the GNN run 
3. Run the GNN and obtain embeddings:
   1. for the first run of the monte carlo, the GNN builds the graph and that takes longer compared to the subsequent runs, which can re-use the constructed graph.
4. Use the embeddings and run xgboost with hyper-parameter optimization and evaluate the quality of embeddings

### 3.2 Running all at once

We have created a script that ties everything in this folder and runs until it generates the best embeddings. If you like to run the components of the code one-by-one, then skip this section and go to the next section 
```
    bash train_all.sh
```
The run takes ~1 day to complete depending on the computer configuration. The log for the 4 steps are available as 0.log, 1.log, 2.log, and 3.log. The trained models and embeddings corresponding to the best validation score will be placed in the folder ./best_embedding/


### 3.3 Running step-by-step

- Step 1:
    Generate GNN-based embeddings for graph that uses edges that have is_clicked=1 for training

```        
bash 0_run_monte_carlo_is_clicked_train.sh
```
- Step 2:
    Pick the best embedding for is_clicked and save it in ./best_embedding/gnn_emb_is_clicked.csv.gz

```
bash 1_extract_results.sh
```

- Step 3: 
    Generate GNN-based embeddings for graph that uses edges that have is_installed=1 for training

```
bash 2_run_monte_carlo_is_installed_train.sh
```

- Step 4:
    Pick the best embedding for is_clicked and save it in ./best_embedding/gnn_emb_is_installed.csv.gz

```
bash 3_extract_results.sh
```