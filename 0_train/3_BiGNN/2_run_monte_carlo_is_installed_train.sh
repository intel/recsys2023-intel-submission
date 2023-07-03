#!/usr/intel/bin/bash

num_sim=10

role1="f_6" 
role2="f_2 f_4 f_16"

mkdir -p graph_data
mkdir -p graph_data/sym_recsys_hetero_CSVDatasets
cp yaml_store/meta.yaml graph_data/sym_recsys_hetero_CSVDatasets/

echo starting $role1 $role2 ...

cp yaml_store/recsys2graph_starter.yaml yaml_store/recsys2graph.yaml
time python convert_tabular_to_graph_data_train.py -r1 ${role1} -r2 ${role2}
role1="${role1// /_}"
role2="${role2// /_}"
echo $role1
echo $role2
cp ./yaml_store/recsys2graph.yaml gnn/configs/recsys2graph.yaml
cd gnn
time ./run-workflow_train.sh ./configs/workflow-config-recsys.yaml
cd .. 
time python xgboost_use_embeddings_wHPO.py | tee -i xgb_isinstalled_sanitized_r1_${role1}_r2_${role2}_trial_0.log
# mv graph_data/tabular_with_gnn_emb.csv.gz graph_data/tabular_with_gnn_emb_is_installed_sanitized_trial_0.csv.gz
mv graph_data/model_graphsage_2L_64.pt graph_data/install_model_graphsage_2L_64_trial_0.pt
mv graph_data/node_emb.pt graph_data/install_node_emb_trial_0.pt

for (( i=1; i<${num_sim}; i++ ));
do
   cd gnn
   time ./run-workflow_train.sh ./configs/workflow-config-recsys-nobuildgraph.yaml
   cd .. 
   time python xgboost_use_embeddings_wHPO.py | tee -i xgb_isinstalled_sanitized_r1_${role1}_r2_${role2}_trial_${i}.log
   # mv graph_data/tabular_with_gnn_emb.csv.gz graph_data/tabular_with_gnn_emb_is_installed_sanitized_trial_${i}.csv.gz
   mv graph_data/model_graphsage_2L_64.pt graph_data/install_model_graphsage_2L_64_trial_${i}.pt
   mv graph_data/node_emb.pt graph_data/install_node_emb_trial_${i}.pt
done
