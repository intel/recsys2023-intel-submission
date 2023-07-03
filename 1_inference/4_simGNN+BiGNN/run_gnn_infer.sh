#!/bin/bash

# set ExpFolder to the best of the GNN training experiments
#ExpFolder="Exp1-2023-06-27-22-16"
ExpFolder="../../0_train/4_simGNN/Exp1-2023-06-27-22-16"
# set best_run_index to the best of the GNN training runs
best_run_index="1"
graph="release_graph"
cuda_dev=0


echo ${ExpFolder}
echo ${graph}
echo ${cuda_dev}


train_graph="../../data/4_simGNN/${graph}_trainval"
train_test_graph="../../data/4_simGNN/${graph}_full"

echo "Run GNN inference"
model_name="3L_SAGE_45-66_${graph}_run_${best_run_index}.pt"
emb_name="node_emb_${graph}_run_${best_run_index}.pt"
CUDA_VISIBLE_DEVICES=${cuda_dev} python -u node_classification_inductive_infer.py --train_graph ${train_graph} --train_test_graph ${train_test_graph} --model_out ${ExpFolder}/${model_name} --nemb_out ${ExpFolder}/${emb_name} |& tee -a ${ExpFolder}/gnn_log.txt
    
echo "map emb to output file in parquet format"
output_file="recsys_data45_67_w_gnn_emb_run_${best_run_index}.parquet"
python -u map_emb_single_recsys_parquet.py --processed_data_path "./recsys_data45_67.csv" --emb_path ${ExpFolder}/${emb_name} --out_data_path ${ExpFolder}/${output_file} |& tee -a ${ExpFolder}/mapsave_log.txt
