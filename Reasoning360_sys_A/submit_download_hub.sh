#!/bin/bash

#echo " I am only this..!"

#echo "Job submitted A100"
#sbatch ./Prompts_System/run_prompts_A100.sh


echo "Downloading DeepSeek LLM"
sbatch ./scripts_data_etc/download_hub_Deep_seek.sh

