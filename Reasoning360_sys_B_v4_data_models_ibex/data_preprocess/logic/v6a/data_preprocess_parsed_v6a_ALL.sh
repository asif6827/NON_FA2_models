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
#python data_spliting_panther.py


#python ./data_preprocess/logic/v6a/our_pre_process_zebrapuzzle_to_guru_parsed_v6a_STST.py --data_path '/export/home/asifali/HF_cache/ZebraLogic' --data_setting 'small_train_small_test' --output_dir '/export/home/asifali/HF_cache/ZebraPuzzle_to_guru_parsed_v6a_STST'

#python ./data_preprocess/logic/v6a/our_pre_process_zebrapuzzle_to_guru_parsed_v6a_MTMT.py --data_path '/export/home/asifali/HF_cache/ZebraLogic' --data_setting 'med_train_med_test' --output_dir '/export/home/asifali/HF_cache/ZebraPuzzle_to_guru_parsed_v6a_MTMT'

#python ./data_preprocess/logic/v6a/our_pre_process_zebrapuzzle_to_guru_parsed_v6a_LTLT.py --data_path '/export/home/asifali/HF_cache/ZebraLogic' --data_setting 'large_train_large_test' --output_dir '/export/home/asifali/HF_cache/ZebraPuzzle_to_guru_parsed_v6a_LTLT'

#python ./data_preprocess/logic/v6a/our_pre_process_zebrapuzzle_to_guru_parsed_v6a_XTXT.py --data_path '/export/home/asifali/HF_cache/ZebraLogic' --data_setting 'xl_train_xl_test' --output_dir '/export/home/asifali/HF_cache/ZebraPuzzle_to_guru_parsed_v6a_XTXT'

#python ./data_preprocess/logic/v6a/our_pre_process_zebrapuzzle_to_guru_parsed_v6a_MLT_MLT.py --data_path '/export/home/asifali/HF_cache/ZebraLogic' --data_setting 'ml_train_ml_test' --output_dir '/export/home/asifali/HF_cache/ZebraPuzzle_to_guru_parsed_v6a_MLT_MLT'

#python ./data_preprocess/logic/v6a/our_pre_process_zebrapuzzle_to_guru_parsed_v6a_MLXL.py --data_path '/export/home/rsaparkhan/HF_cache/ZebraLogic' --data_setting 'mlxl_train_mlxl_test' --output_dir '/export/home/rsaparkhan/HF_cache/ZebraPuzzle_to_guru_parsed_v6a_MLXL'


python ./data_preprocess/logic/v6a/our_pre_process_zebrapuzzle_to_guru_parsed_v6a_MLXL.py --data_path '/export/home/asifali/HF_cache/ZebraLogic' --data_setting 'mlxl_train_mlxl_test' --output_dir '/export/home/asifali/HF_cache/ZebraPuzzle_to_guru_parsed_v6a_MLXL'

python ./data_preprocess/logic/v6a/our_pre_process_zebrapuzzle_to_guru_parsed_v6a_MLXL_Llama.py --data_path '/export/home/asifali/HF_cache/ZebraLogic' --data_setting 'mlxl_train_mlxl_test' --output_dir '/export/home/asifali/HF_cache/ZebraPuzzle_to_guru_parsed_v6a_MLXL'

python ./data_preprocess/logic/v6a/our_pre_process_zebrapuzzle_to_guru_parsed_v6a_MLXL_Llama_v2.py --data_path '/export/home/asifali/HF_cache/ZebraLogic' --data_setting 'mlxl_train_mlxl_test' --output_dir '/export/home/asifali/HF_cache/ZebraPuzzle_to_guru_parsed_v6a_MLXL'

python ./data_preprocess/logic/v6a/our_pre_process_zebrapuzzle_to_guru_parsed_v6a_MLXL_DeepSeek.py --data_path '/export/home/asifali/HF_cache/ZebraLogic' --data_setting 'mlxl_train_mlxl_test' --output_dir '/export/home/asifali/HF_cache/ZebraPuzzle_to_guru_parsed_v6a_MLXL'

python ./data_preprocess/logic/v6a/our_pre_process_zebrapuzzle_to_guru_parsed_v6a_MLXL_Phi4.py --data_path '/export/home/asifali/HF_cache/ZebraLogic' --data_setting 'mlxl_train_mlxl_test' --output_dir '/export/home/asifali/HF_cache/ZebraPuzzle_to_guru_parsed_v6a_MLXL'

python ./data_preprocess/logic/v6a/our_pre_process_zebrapuzzle_to_guru_parsed_v6a_MLXL_EXAONE.py --data_path '/export/home/asifali/HF_cache/ZebraLogic' --data_setting 'mlxl_train_mlxl_test' --output_dir '/export/home/asifali/HF_cache/ZebraPuzzle_to_guru_parsed_v6a_MLXL'

python ./data_preprocess/logic/v6a/our_pre_process_zebrapuzzle_to_guru_parsed_v6a_MLXL_SmolLM3.py --data_path '/export/home/asifali/HF_cache/ZebraLogic' --data_setting 'mlxl_train_mlxl_test' --output_dir '/export/home/asifali/HF_cache/ZebraPuzzle_to_guru_parsed_v6a_MLXL'

python ./data_preprocess/logic/v6a/our_pre_process_zebrapuzzle_to_guru_parsed_v6a_MLXL_Gemma3.py --data_path '/export/home/asifali/HF_cache/ZebraLogic' --data_setting 'mlxl_train_mlxl_test' --output_dir '/export/home/asifali/HF_cache/ZebraPuzzle_to_guru_parsed_v6a_MLXL'

python ./data_preprocess/logic/v6a/our_pre_process_zebrapuzzle_to_guru_parsed_v6a_MLXL_Granite3.py --data_path '/export/home/asifali/HF_cache/ZebraLogic' --data_setting 'mlxl_train_mlxl_test' --output_dir '/export/home/asifali/HF_cache/ZebraPuzzle_to_guru_parsed_v6a_MLXL'

#python ./data_preprocess/logic/v6a/our_pre_process_zebrapuzzle_to_guru_parsed_v6a_ZTZT.py --data_path '/export/home/asifali/HF_cache/ZebraLogic' --data_setting 'zl_train_zl_test' --output_dir '/export/home/asifali/HF_cache/ZebraPuzzle_to_guru_parsed_v6a_ZTZT'

#python ./data_preprocess/logic/our_pre_process_guru_readjust_parsed_v2.py --input_path '/export/home/asifali/HF_cache/guru_data' --output_dir '/export/home/asifali/HF_cache/guru_data_adjusted_parsed_v2'




### Testing in HP computer

# --data_path '/home/asif/data3/HF_cache/ZebraLogic/' --data_setting 'small_train_small_test' --shot 'one' --output_dir '/home/asif/data3/HF_cache/ZebraPuzzle_to_guru_1shot'

# args.output_dir = '/export/home/asifali/HF_cache/ZebraPuzzle_to_guru'


nvidia-smi