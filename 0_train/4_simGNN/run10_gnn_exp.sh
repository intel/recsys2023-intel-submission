#!/bin/bash

##CHANGE THESE BEFORE RUNNING
num_runs=10
num_epochs=10


ExpName="Exp1"
graph="release_graph"
fan_out="5,5,5"
cuda_dev=0


currTime=$(date +%Y-%m-%d-%H-%M)
ExpFolder="${ExpName}-${currTime}"
echo ${ExpFolder}
echo ${graph}
echo ${fan_out}
echo ${cuda_dev}


train_graph="../../data/4_simGNN/${graph}_trainval"


mkdir -p ${ExpFolder}
for ((i=1; i<=$num_runs; i++)); do
    #echo "Run GNN training"
    model_name="3L_SAGE_45-66_${graph}_run_${i}.pt"
    emb_name="node_emb_${graph}_run_${i}.pt"
    CUDA_VISIBLE_DEVICES=${cuda_dev} python -u node_classification_inductive_train.py --train_graph ${train_graph} --model_out ./${ExpFolder}/${model_name} --nemb_out ./${ExpFolder}/${emb_name} --num_epochs ${num_epochs} --fan_out ${fan_out} |& tee -a ./${ExpFolder}/gnn_log.txt
done

