#!/bin/bash

#echo " I am only this..!"

#echo "Job submitted A100"
#sbatch ./Prompts_System/run_prompts_A100.sh


#echo "Downloading LLM"
#sbatch ./scripts_data_etc/download_hub.sh

#echo "Data Pre-Processing Runing"
#sbatch ./data_preprocess/logic/data_preprocess.sh


#echo "Data Pre-Processing Runing"
#sbatch ./data_preprocess/logic/data_preprocess_parsed.sh


#echo "Data Pre-Processing Runing"
#sbatch ./data_preprocess/logic/data_preprocess_parsed_v2.sh

#echo "Data Pre-Processing Runing"
#sbatch ./data_preprocess/logic/data_preprocess_parsed_v4.sh

#echo "Data Pre-Processing Runing"
#sbatch ./data_preprocess/logic/data_preprocess_parsed_v5.sh

#echo "Data Pre-Processing Runing"
#sbatch ./data_preprocess/logic/data_preprocess_parsed_v5a.sh

#echo "Data Pre-Processing Runing"
#sbatch ./data_preprocess/logic/data_preprocess_parsed_v5b.sh

#echo "Data Pre-Processing Runing"
#sbatch ./data_preprocess/logic/data_preprocess_parsed_v5c.sh


#echo "Data Pre-Processing Runing"
#sbatch ./data_preprocess/logic/data_preprocess_parsed_v5d_STST.sh


#echo "Data Pre-Processing Runing"
#sbatch ./data_preprocess/logic/data_preprocess_parsed_v5e_STST.sh



#echo "Data Pre-Processing Runing"
#sbatch ./data_preprocess/logic/data_preprocess_parsed_v5g_STST.sh


#echo "Data Pre-Processing Runing"
#sbatch ./data_preprocess/logic/data_preprocess_parsed_v5h_STST.sh


#echo "Data Pre-Processing Runing"
#sbatch ./data_preprocess/logic/data_preprocess_parsed_v5i_STST.sh


#echo "Data Pre-Processing Runing"
#sbatch ./data_preprocess/logic/data_preprocess_parsed_v5j_STST.sh



#echo "Data Pre-Processing Runing"
#sbatch ./data_preprocess/logic/data_preprocess_parsed_v5d_MTMT.sh


#echo "Data Pre-Processing Runing"
#sbatch ./data_preprocess/logic/data_preprocess_parsed_v6.sh



#echo "Data Pre-Processing Runing"
#sbatch ./data_preprocess/logic/v6a/data_preprocess_parsed_v6a_ALL.sh




#echo "Data Pre-Processing Runing"
#sbatch ./data_preprocess/logic/data_preprocess_parsed_v7.sh


#echo "Data spliting code Runing"
#sbatch ./scripts_data_etc/data_spliting.sh


#echo "Job submitted H200"
#sbatch ./scripts/train/RL_testing_guru.sh


#echo "Job submitted A100"
#sbatch ./scripts/train/RL_testing_zebra_A100.sh

#echo "Job submitted H200"
#sbatch ./scripts/train/RL_testing_zebra_H200.sh


#echo "Job submitted GPU-ALL"
#sbatch ./scripts/train/RL_testing_zebra_GPU_ALL.sh


#echo "Job submitted H200"
#sbatch ./scripts/train/RL_testing_real.sh


#echo "Job submitted H200"
#sbatch ./scripts/train/guru_rl_qwen25_1_5b_fsdp_logic_H200.sh


#echo "Job submitted H200 STMT"
#sbatch ./scripts/train/zebra_rl_fsdp_logic_H200_STMT_1_1shot.sh



#echo "Job submitted A100 STST"
#sbatch ./scripts/train/STST_1_A100_zebra_rl_fsdp.sh


#####echo "Job submitted A100 STST"
#####sbatch ./scripts/train/zebra_rl_fsdp_logic_A100_STST_1_1shot.sh


#echo "Job submitted H200 STST"
#sbatch ./scripts/train/STST_1_1shot_H200_zebra_rl_fsdp.sh


#echo "Job submitted H200"
#sbatch ./scripts/train/zebra_rl_fsdp_logic_H200_STMT_1.sh


#sbatch ./scripts/train/Natural_Language/zebra_rl_fsdp_logic_A100_STST_4.sh

#sbatch ./scripts/train/Natural_Language/zebra_rl_fsdp_logic_A100_STST_4.sh


#sbatch ./scripts/train/singlenode_rl_qwen25_1_5b_fsdp_3K_test.sh


#sbatch ./scripts_Qwen2_5-1_5B_Instruct/ray_eval_singlenode_Qwen2_5-1_5B_Instruct_20_samples.sh

#echo "Job submitted GPU-ALL"
#sbatch ./Prompts_System/run_prompts_GPU_ALL.sh

#echo "Job submitted GPU-ALL"
#SLURM_SCRIPT_H200="./run_prompts_A100.sh"

#echo "Job submitted GPU-ALL"
#SLURM_SCRIPT_H200="./run_prompts_H200.sh"

#echo "Job submitted GPU-ALL"
#SLURM_SCRIPT_H200="./run_prompts_GPU_ALL.sh"



#if false; then
ACRONYM="MTMT"
echo "Submitting Training job H200 + GT"
TRAIN_TEMP_LIST=(0.8)
TEST_TEMP_LIST=(0.0)
SCORING_LIST=("gt")
EPOCH_LIST=(20)
TEST_LIST=(3)
ACC_W_LIST=(0.8)
Z3_W_LIST=(0.2)
SWITCH_EPOCH_LIST=(80)
SYSTEM_NAME_LIST=("Reasoning360_sys_B_v31")
EVAL_PATH_LIST=("med_train_med_test_1_parsed_v6a_${ACRONYM}")
DATA_PATH_LIST=("ZebraPuzzle_to_guru_parsed_v6a_${ACRONYM}/med_train_med_test")



SLURM_SCRIPT_H200="./scripts_Qwen3_4B/train/Parsed_v2/${ACRONYM}_1_1shot_H200_zebra_rl_fsdp_parsed_v2.sh"
for i in "${!TRAIN_TEMP_LIST[@]}"; do
    TRAIN_TEMP=${TRAIN_TEMP_LIST[$i]}
    TEST_TEMP=${TEST_TEMP_LIST[$i]}
    SCORING_METHOD=${SCORING_LIST[$i]}
    EPOCH=${EPOCH_LIST[$i]}
    TEST_FREQUENCY=${TEST_LIST[$i]}
    ACC_W=${ACC_W_LIST[$i]}
    Z3_W=${Z3_W_LIST[$i]}
    SWITCH_EPOCH=${SWITCH_EPOCH_LIST[$i]}
    SYSTEM_NAME=${SYSTEM_NAME_LIST[$i]}
    EVAL_PATH=${EVAL_PATH_LIST[$i]}
    DATA_PATH=${DATA_PATH_LIST[$i]}

    echo "Submitting job: TRAIN-TEMP=$TRAIN_TEMP, TEST-TEMP=$TEST_TEMP, SCORING-METHOD=$SCORING_METHOD, EPOCH=$EPOCH, TEST-FREQUENCY=$TEST_FREQUENCY,
    ACC_W=$ACC_W, Z3_W=$Z3_W, EPOCH_SWITCH=$SWITCH_EPOCH SYSTEM_NAME=$SYSTEM_NAME EVAL_PATH=$EVAL_PATH DATA_PATH=$DATA_PATH"
    sbatch $SLURM_SCRIPT_H200 $TRAIN_TEMP $TEST_TEMP $SCORING_METHOD $EPOCH $TEST_FREQUENCY $ACC_W $Z3_W $SWITCH_EPOCH $SYSTEM_NAME $EVAL_PATH $DATA_PATH
done
echo "All jobs submitted H200."
#fi




if false; then
    ACRONYM="MTMT"
    echo "Submitting Training job A100 + GT"
    TRAIN_TEMP_LIST=(0.6)
    TEST_TEMP_LIST=(0.0)
    SCORING_LIST=("gt")
    EPOCH_LIST=(100)
    TEST_LIST=(4)
    ACC_W_LIST=(0.8)
    Z3_W_LIST=(0.2)
    SWITCH_EPOCH_LIST=(80)
    SYSTEM_NAME_LIST=("Reasoning360_sys_B_v1")
    EVAL_PATH_LIST=("med_train_med_test_1_parsed_v6a_${ACRONYM}")
    DATA_PATH_LIST=("ZebraPuzzle_to_guru_parsed_v6a_${ACRONYM}/med_train_med_test")



    SLURM_SCRIPT_H200="./scripts_Qwen3_4B/train/Parsed_v2/${ACRONYM}_1_1shot_A100_zebra_rl_fsdp_parsed_v2.sh"
    for i in "${!TRAIN_TEMP_LIST[@]}"; do
        TRAIN_TEMP=${TRAIN_TEMP_LIST[$i]}
        TEST_TEMP=${TEST_TEMP_LIST[$i]}
        SCORING_METHOD=${SCORING_LIST[$i]}
        EPOCH=${EPOCH_LIST[$i]}
        TEST_FREQUENCY=${TEST_LIST[$i]}
        ACC_W=${ACC_W_LIST[$i]}
        Z3_W=${Z3_W_LIST[$i]}
        SWITCH_EPOCH=${SWITCH_EPOCH_LIST[$i]}
        SYSTEM_NAME=${SYSTEM_NAME_LIST[$i]}
        EVAL_PATH=${EVAL_PATH_LIST[$i]}
        DATA_PATH=${DATA_PATH_LIST[$i]}

        echo "Submitting job: TRAIN-TEMP=$TRAIN_TEMP, TEST-TEMP=$TEST_TEMP, SCORING-METHOD=$SCORING_METHOD, EPOCH=$EPOCH, TEST-FREQUENCY=$TEST_FREQUENCY,
        ACC_W=$ACC_W, Z3_W=$Z3_W, EPOCH_SWITCH=$SWITCH_EPOCH SYSTEM_NAME=$SYSTEM_NAME EVAL_PATH=$EVAL_PATH DATA_PATH=$DATA_PATH"
        sbatch $SLURM_SCRIPT_H200 $TRAIN_TEMP $TEST_TEMP $SCORING_METHOD $EPOCH $TEST_FREQUENCY $ACC_W $Z3_W $SWITCH_EPOCH $SYSTEM_NAME $EVAL_PATH $DATA_PATH
    done
    echo "All jobs submitted A100."
fi


















































