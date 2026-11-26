#!/bin/bash -l

#SBATCH -J Data-Processing #job name
#SBATCH -p gpu-all # queue used
#SBATCH --gres gpu:1 #number of gpus needed, default is 1
#SBATCH -c 8  #number of CPUs needed, default is 1 
#SBATCH --mem 32GB #amount of memory needed, default
#SBATCH --output=/export/alt-insight/NON_FA2_models/Reasoning360_sys_B_v29_case5_from_Chkpt_mraza/all_logs/%j-%x-slurm.out
#SBATCH --error=/export/alt-insight/NON_FA2_models/Reasoning360_sys_B_v29_case5_from_Chkpt_mraza/all_logs/%j-%x-slurm.err
#SBATCH --mail-user=asif6827@gmail.com


module load cuda12.4/toolkit

nvidia-smi
export TRANSFORMERS_CACHE="/export/alt-insight/HF_cache"
export HF_HOME="/export/alt-insight/HF_cache"
export HF_DATASETS_CACHE="/export/alt-insight/HF_cache"

source activate Reasoning360
#python data_spliting_panther.py


python ./data_preprocess/logic/our_pre_process_zebrapuzzle_to_guru_parsed.py --data_path '/export/alt-insight/HF_cache/ZebraLogic' --data_setting 'small_train_small_test' --output_dir '/export/alt-insight/HF_cache/ZebraPuzzle_to_guru_parsed'

python ./data_preprocess/logic/our_pre_process_guru_readjust_parsed.py --input_path '/export/alt-insight/HF_cache/guru_data' --output_dir '/export/alt-insight/HF_cache/guru_data_adjusted_parsed'




### Testing in HP computer

# --data_path '/export/alt-insight/HF_cache/ZebraLogic/' --data_setting 'small_train_small_test' --shot 'one' --output_dir '/export/alt-insight/HF_cache/ZebraPuzzle_to_guru_1shot'

# args.output_dir = '/export/alt-insight/HF_cache/ZebraPuzzle_to_guru'


nvidia-smi