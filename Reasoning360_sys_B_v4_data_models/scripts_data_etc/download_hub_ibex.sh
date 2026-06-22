#!/bin/bash -l

#!/bin/bash
#SBATCH --time=6-05:30:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=v100:1
#SBATCH --cpus-per-gpu=1
#SBATCH --mem=100G
#SBATCH --partition=batch
#SBATCH --job-name=download_qwen3
#SBATCH --mail-type=ALL
#SBATCH --output=%x-%j-slurm.out
#SBATCH --error=%x-%j-slurm.err


nvidia-smi
export TRANSFORMERS_CACHE="/ibex/scratch/zakroum/lab-asif/HF_cache"
export HF_HOME="/ibex/scratch/zakroum/lab-asif/HF_cache"
export HF_DATASETS_CACHE="/ibex/scratch/zakroum/lab-asif/HF_cache"

source activate zebrapuzzles
python ./scripts_data_etc/download_hub_Qwen3_ibex.py
nvidia-smi