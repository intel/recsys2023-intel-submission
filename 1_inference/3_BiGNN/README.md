# Instructions for running the self-supervised GNN for RecSys2023

## 1. Introduction:
### 1.1 Modeling as a graph
    This section of the code models the dataset into a graph of two types of nodes -- role-1 and role-2. Each impression of the dataset is modelled as an edge between a node of type role-1 and a node of type role-2. 

    Since we have no information as what features belong users and advertisements (we use the term 'role' to refer to either of them), we ran experiments to try out various feature-to-role assignments and measuring the performance on the validation dataset. Finally, we settled with the following assignment
        - Role 1: f_6
        - Role 2: f_2, f_4, and f_16

    It is important to note that these roles need not refer to users and advertisements individually. We only expect them to refer to groups of users and groups of advertisements. 

    Therefore, every impression connects a node of type role 1 and node of type role 2. 

### 1.1 Summary
    The graph is modeled with nodes corresponding to roles 1 and 2 and edges corresponding to every impression

## 2. Setup:
    Place the Sharechat Recsys2023 datasets in the folder $GIT_ROOT/dataset/sharechat_recsys2023_data
        a. $GIT_ROOT/data/sharechat_recsys2023_data/train/ --> place the training dataset here
        b. $GIT_ROOT/data/sharechat_recsys2023_data/test/ --> place the test dataset here

## 3. Run instructions
    1. Place the trained models and embeddings in ./best_embedding/
    2. execute the run_inference.sh script
```
bash run_inference.sh
```
## 4. Flow

    This flow reads the trained model from the 0_train/3_BiGNN and runs inference to generate two sets of embeddings
    - best_embedding/gnn_emb_is_clicked.csv.gz
    - best_embedding/gnn_emb_is_installed.csv.gz
