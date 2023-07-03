#!/usr/intel/bin/bash


role1="f_6" 
role2="f_2 f_4 f_16"

mkdir -p graph_data
mkdir -p graph_data/sym_recsys_hetero_CSVDatasets
cp yaml_store/meta.yaml graph_data/sym_recsys_hetero_CSVDatasets/

echo starting $role1 $role2 ...

cp yaml_store/recsys2graph_starter_isclicked.yaml yaml_store/recsys2graph.yaml
time python convert_tabular_to_graph_data.py -r1 ${role1} -r2 ${role2}
role1="${role1// /_}"
role2="${role2// /_}"
echo $role1
echo $role2
cp ./yaml_store/recsys2graph.yaml gnn/configs/recsys2graph.yaml
cp best_embedding/model_graphsage_2L_64_isclicked.pt best_embedding/model_graphsage_2L_64.pt
cp best_embedding/node_emb_isclicked.pt best_embedding/node_emb.pt
cd gnn
time ./run-workflow_inference.sh ./configs/workflow-config-recsys.yaml
cd .. 
mv graph_data/tabular_with_gnn_emb.csv.gz best_embedding/gnn_emb_is_clicked.csv.gz

cp yaml_store/recsys2graph_starter.yaml yaml_store/recsys2graph.yaml
time python convert_tabular_to_graph_data.py -r1 ${role1} -r2 ${role2}
role1="${role1// /_}"
role2="${role2// /_}"
echo $role1
echo $role2
cp ./yaml_store/recsys2graph.yaml gnn/configs/recsys2graph.yaml
cp best_embedding/model_graphsage_2L_64_isinstalled.pt best_embedding/model_graphsage_2L_64.pt
cp best_embedding/node_emb_isinstalled.pt best_embedding/node_emb.pt
cd gnn
time ./run-workflow_inference.sh ./configs/workflow-config-recsys.yaml
cd .. 
mv graph_data/tabular_with_gnn_emb.csv.gz best_embedding/gnn_emb_is_installed.csv.gz

