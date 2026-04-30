#!/bin/bash

#echo " I am only this..!"


#echo "Data Pre-Processing Runing"
#sbatch ./data_preprocess/logic/v6a/data_preprocess_parsed_v6a_ALL.sh


echo "Model Downloading...!"
batch ./scripts_data_etc/download_hub_qwen3.sh


#echo "Model Downloading...!"
#sbatch ./scripts_data_etc/download_hub_ibex.sh



#cho "Data Pre-Processing Runing"
#sbatch ./data_preprocess/logic/data_preprocess_parsed_v7.sh

