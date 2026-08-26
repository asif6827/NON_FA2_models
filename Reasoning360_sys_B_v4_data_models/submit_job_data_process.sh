#!/bin/bash

#echo " I am only this..!"


mkdir -p ./all_logs

#echo "Data Processing script Running..!"
#sbatch ./data_preprocess/logic/v6a/data_preprocess_parsed_v6a_ALL.sh


echo "Data Processing script Running..!"
sbatch ./data_preprocess/logic/v7/data_preprocess_parsed_v7.sh


#echo "Data Pre-Processing Runing"
#sbatch ./data_preprocess/logic/v6a/data_preprocess_parsed_v6a_ALL.sh