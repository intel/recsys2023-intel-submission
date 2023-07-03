#!/usr/intel/bin/bash
# set -xv
array=()
for file in xgb_isinstalled_sanitized_r1_f_6_r2_f_2_f_4_f_16*log;
do 
    #echo $file
    grouping1=`sed "/\[999\]/q" ${file} | grep "logloss" | sed "s/^.*val-logloss://" | sed "s/^.*validation-logloss://" | sed "s/\s.*//g" | sort -n | head -n 1`
    grouping2=`awk "/\[999\]/{y=1;next}y" ${file} | grep "logloss" | sed "s/^.*val-logloss://" | sed "s/^.*validation-logloss://" | sed "s/\s.*//g" | sort -n | head -n 1`
    group_name1=`echo ${file} | sed "s/^.*r1_//g" | sed "s/_r2.*//g"`
    group_name2=`echo ${file} | sed "s/^.*r2_//g" | sed "s/.log//g"`
    #echo metric = ${grouping1}
    array+=($grouping2)
    #echo $group_name1 $group_name2 $grouping2
done
best_idx=`echo "${array[@]}" | tr -s ' ' '\n' | awk '{print($0" "NR)}' |sort -g -k1,1 | head -1 | cut -f2 -d' '`
best_idx=$(($best_idx-1))
echo $best_idx
mkdir -p best_embedding
rm -rf best_embedding/gnn_emb_is_installed.csv.gz
rm -rf best_embedding/model_graphsage_2L_64_isinstalled.pt
rm -rf best_embedding/node_emb_isinstalled.pt

echo "Found best trial to be ${best_idx} and stored it in best_embedding/"
# ln -s graph_data/tabular_with_gnn_emb_is_installed_sanitized_trial_${best_idx}.csv.gz best_embedding/gnn_emb_is_installed.csv.gz
cd best_embedding
cp ../graph_data/install_model_graphsage_2L_64_trial_${best_idx}.pt model_graphsage_2L_64_isinstalled.pt
cp ../graph_data/install_node_emb_trial_${best_idx}.pt node_emb_isinstalled.pt