#!/bin/bash

#echo " I am only this..!"


#echo "Data Pre-Processing Runing"
#sbatch ./data_preprocess/logic/v6a/data_preprocess_parsed_v6a_ALL.sh


echo "Model Gemmas Downloading...!"

#sbatch ./scripts_data_etc/download_hub_SmolLM3_3B.sh
sbatch ./scripts_data_etc/download_hub_gemma.sh
#sbatch ./scripts_data_etc/download_hub_Phi4.sh
#sbatch ./scripts_data_etc/download_hub_llama.sh
#sbatch ./scripts_data_etc/download_hub_EXAONE.sh


#echo "Model Downloading...!"
#sbatch ./scripts_data_etc/download_hub_ibex.sh



#cho "Data Pre-Processing Runing"
#sbatch ./data_preprocess/logic/data_preprocess_parsed_v7.sh

