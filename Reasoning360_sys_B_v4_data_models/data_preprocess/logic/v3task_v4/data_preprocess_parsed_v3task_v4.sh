#!/bin/bash -l

#SBATCH -J Data-Processing #job name
#SBATCH -p gpu-all # queue used
#SBATCH --gres gpu:1 #number of gpus needed, default is 1
#SBATCH -c 8  #number of CPUs needed, default is 1 
#SBATCH --mem 32GB #amount of memory needed, default
#SBATCH --output=./all_logs/%j-%x-slurm.out
#SBATCH --error=./all_logs/%j-%x-slurm.err
#SBATCH --mail-user=asif6827@gmail.com


set -e

module load cuda12.4/toolkit

nvidia-smi
export TRANSFORMERS_CACHE="/export/alt-insight/HF_cache"
export HF_HOME="/export/alt-insight/HF_cache"
export HF_DATASETS_CACHE="/export/alt-insight/HF_cache"

source activate zebrapuzzles


python ./data_preprocess/logic/v3task_v4/our_pre_process_zebrapuzzle_to_guru_parsed_v3task_v4_MLXL.py --data_path '/export/alt-insight/HF_cache/ZebraLogic' --data_setting 'mlxl_train_mlxl_test' --output_dir '/export/alt-insight/HF_cache/ZebraPuzzle_to_guru_parsed_v3task_v4_MLXL'



nvidia-smi
