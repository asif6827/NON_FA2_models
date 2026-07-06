#!/bin/bash -l

#SBATCH -J Data-Processing-ARLSAT #job name
#SBATCH -p gpu-all # queue used
#SBATCH --gres gpu:1 #number of gpus needed, default is 1
#SBATCH -c 8  #number of CPUs needed, default is 1 
#SBATCH --mem 32GB #amount of memory needed, default
#SBATCH --output=./all_logs/%j-%x-slurm.out
#SBATCH --error=./all_logs/%j-%x-slurm.err
#SBATCH --mail-user=asif6827@gmail.com


module load cuda12.4/toolkit

nvidia-smi
export TRANSFORMERS_CACHE="/export/home/asifali/HF_cache"
export HF_HOME="/export/home/asifali/HF_cache"
export HF_DATASETS_CACHE="/export/home/asifali/HF_cache"

source activate zebrapuzzles
#python data_spliting_panther.py



#python ./data_preprocess/logic/v6a/AR_LSAT_parsed_v6a_MLXL_ordering.py --data_path '/export/home/asifali/HF_cache/AR_LSAT' --data_setting 'mlxl_train_mlxl_test' --output_dir '/export/home/asifali/HF_cache/AR_LSAT_ordering_v6a_MLXL'

#python ./data_preprocess/logic/v6a/AR_LSAT_parsed_v6a_MLXL_grouping.py --data_path '/export/home/asifali/HF_cache/AR_LSAT' --data_setting 'mlxl_train_mlxl_test' --output_dir '/export/home/asifali/HF_cache/AR_LSAT_grouping_v6a_MLXL'

#python ./data_preprocess/logic/v6a/AR_LSAT_parsed_v6a_MLXL_assignment.py --data_path '/export/home/asifali/HF_cache/AR_LSAT' --data_setting 'mlxl_train_mlxl_test' --output_dir '/export/home/asifali/HF_cache/AR_LSAT_assignment_v6a_MLXL'


python ./data_preprocess/logic/v6a/AR_LSAT_parsed_v6a_MLXL_ordering_phi4.py --data_path '/export/home/asifali/HF_cache/AR_LSAT' --data_setting 'mlxl_train_mlxl_test' --output_dir '/export/home/asifali/HF_cache/AR_LSAT_ordering_v6a_MLXL'

python ./data_preprocess/logic/v6a/AR_LSAT_parsed_v6a_MLXL_grouping_phi4.py --data_path '/export/home/asifali/HF_cache/AR_LSAT' --data_setting 'mlxl_train_mlxl_test' --output_dir '/export/home/asifali/HF_cache/AR_LSAT_grouping_v6a_MLXL'

python ./data_preprocess/logic/v6a/AR_LSAT_parsed_v6a_MLXL_assignment_phi4.py --data_path '/export/home/asifali/HF_cache/AR_LSAT' --data_setting 'mlxl_train_mlxl_test' --output_dir '/export/home/asifali/HF_cache/AR_LSAT_assignment_v6a_MLXL'



python ./data_preprocess/logic/v6a/AR_LSAT_parsed_v6a_A_MLXL_ordering.py --data_path '/export/home/asifali/HF_cache/AR_LSAT' --data_setting 'mlxl_train_mlxl_test' --output_dir '/export/home/asifali/HF_cache/AR_LSAT_ordering_A_v6a_MLXL'

python ./data_preprocess/logic/v6a/AR_LSAT_parsed_v6a_A_MLXL_grouping.py --data_path '/export/home/asifali/HF_cache/AR_LSAT' --data_setting 'mlxl_train_mlxl_test' --output_dir '/export/home/asifali/HF_cache/AR_LSAT_grouping_A_v6a_MLXL'

python ./data_preprocess/logic/v6a/AR_LSAT_parsed_v6a_A_MLXL_assignment.py --data_path '/export/home/asifali/HF_cache/AR_LSAT' --data_setting 'mlxl_train_mlxl_test' --output_dir '/export/home/asifali/HF_cache/AR_LSAT_assignment_A_v6a_MLXL'




#python ./data_preprocess/logic/v6a/AR_LSAT_parsed_v6a_A_MLXL_ordering_phi4.py --data_path '/export/home/asifali/HF_cache/AR_LSAT' --data_setting 'mlxl_train_mlxl_test' --output_dir '/export/home/asifali/HF_cache/AR_LSAT_ordering_A_v6a_MLXL'

#python ./data_preprocess/logic/v6a/AR_LSAT_parsed_v6a_A_MLXL_grouping_phi4.py --data_path '/export/home/asifali/HF_cache/AR_LSAT' --data_setting 'mlxl_train_mlxl_test' --output_dir '/export/home/asifali/HF_cache/AR_LSAT_grouping_A_v6a_MLXL'

#python ./data_preprocess/logic/v6a/AR_LSAT_parsed_v6a_A_MLXL_assignment_phi4.py --data_path '/export/home/asifali/HF_cache/AR_LSAT' --data_setting 'mlxl_train_mlxl_test' --output_dir '/export/home/asifali/HF_cache/AR_LSAT_assignment_A_v6a_MLXL'



nvidia-smi