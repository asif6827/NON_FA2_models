#!/bin/bash -l

#SBATCH -J Data-Processing #job name
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


python ./data_preprocess/logic/v6b/our_pre_process_zebrapuzzle_to_guru_parsed_v6b_MLXL.py --data_path '/export/home/asifali/HF_cache/ZebraLogic' --data_setting 'mlxl_train_mlxl_test' --output_dir '/export/home/asifali/HF_cache/ZebraPuzzle_to_guru_parsed_v6b_MLXL'



nvidia-smi
