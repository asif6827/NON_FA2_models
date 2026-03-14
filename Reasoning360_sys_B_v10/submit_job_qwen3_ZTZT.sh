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





if false; then
    ACRONYM="STST"
    echo "Submitting Training job H200 + GT"
    TRAIN_TEMP_LIST=(0.6)
    TEST_TEMP_LIST=(0.0)
    SCORING_LIST=("gt")
    EPOCH_LIST=(100)
    TEST_LIST=(5)   # Need to check it
    ACC_W_LIST=(0.8)
    Z3_W_LIST=(0.2)
    SWITCH_EPOCH_LIST=(80)
    SYSTEM_NAME_LIST=("Reasoning360_sys_B_v1")
    EVAL_PATH_LIST=("small_train_small_test_1_parsed_v6a_${ACRONYM}")
    DATA_PATH_LIST=("ZebraPuzzle_to_guru_parsed_v6a_${ACRONYM}/small_train_small_test")



    SLURM_SCRIPT_H200="./scripts_Qwen25_15_Instruct/train/Parsed_v2/${ACRONYM}_1_1shot_H200_zebra_rl_fsdp_parsed_v2.sh"
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
fi






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
SYSTEM_NAME_LIST=("Reasoning360_sys_B_v10")
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











































































































































if false; then
    ACRONYM="STST"
    echo "Submitting Training job A100 + GT"
    TRAIN_TEMP_LIST=(0.8)
    TEST_TEMP_LIST=(0.8)
    SCORING_LIST=("gt")
    EPOCH_LIST=(150)
    TEST_LIST=(5)
    ACC_W_LIST=(0.8)
    Z3_W_LIST=(0.2)
    SWITCH_EPOCH_LIST=(80)
    SYSTEM_NAME_LIST=("Reasoning360_sys_B_v3")
    EVAL_PATH_LIST=("small_train_small_test_1_parsed_v5h_${ACRONYM}")
    DATA_PATH_LIST=("ZebraPuzzle_to_guru_parsed_v5h_${ACRONYM}/small_train_small_test")



    SLURM_SCRIPT_A100="./scripts/train/Parsed_v2/${ACRONYM}_1_1shot_A100_zebra_rl_fsdp_parsed_v2.sh"
    for i in "${!TRAIN_TEMP_LIST[@]}"; do
        TRAIN_TEMP=${TRAIN_TEMP_LIST[$i]}
        TEST_TEMP=${TEST_TEMP_LIST[$i]}
        SCORING_METHOD=${SCORING_LIST[0]}
        EPOCH=${EPOCH_LIST[0]}
        TEST_FREQUENCY=${TEST_LIST[0]}
        ACC_W=${ACC_W_LIST[$i]}
        Z3_W=${Z3_W_LIST[$i]}
        SWITCH_EPOCH=${SWITCH_EPOCH_LIST[$i]}
        SYSTEM_NAME=${SYSTEM_NAME_LIST[$i]}
        EVAL_PATH=${EVAL_PATH_LIST[$i]}
        DATA_PATH=${DATA_PATH_LIST[$i]}


        echo "Submitting job: TRAIN-TEMP=$TRAIN_TEMP, TEST-TEMP=$TEST_TEMP, SCORING-METHOD=$SCORING_METHOD, EPOCH=$EPOCH, TEST-FREQUENCY=$TEST_FREQUENCY,
        ACC_W=$ACC_W, Z3_W=$Z3_W, EPOCH_SWITCH=$SWITCH_EPOCH SYSTEM_NAME=$SYSTEM_NAME EVAL_PATH=$EVAL_PATH DATA_PATH=$DATA_PATH"
        sbatch $SLURM_SCRIPT_A100 $TRAIN_TEMP $TEST_TEMP $SCORING_METHOD $EPOCH $TEST_FREQUENCY $ACC_W $Z3_W $SWITCH_EPOCH $SYSTEM_NAME $EVAL_PATH $DATA_PATH
    done

    echo "All jobs submitted A100."
    echo "All scripts finished here."
fi








if false; then
    ACRONYM="STST"
    echo "Submitting Training job A100 + GT"
    TRAIN_TEMP_LIST=(0.8)
    TEST_TEMP_LIST=(0.8)
    SCORING_LIST=("gt")
    EPOCH_LIST=(150)
    TEST_LIST=(5)
    ACC_W_LIST=(0.8)
    Z3_W_LIST=(0.2)
    SWITCH_EPOCH_LIST=(80)
    SYSTEM_NAME_LIST=("Reasoning360_sys_B_v3")
    EVAL_PATH_LIST=("small_train_small_test_1_parsed_v5h_${ACRONYM}")
    DATA_PATH_LIST=("ZebraPuzzle_to_guru_parsed_v5h_${ACRONYM}/small_train_small_test")



    SLURM_SCRIPT_A100="./scripts/train/Parsed_v2/${ACRONYM}_1_1shot_A100_zebra_rl_fsdp_parsed_v2.sh"
    for i in "${!TRAIN_TEMP_LIST[@]}"; do
        TRAIN_TEMP=${TRAIN_TEMP_LIST[$i]}
        TEST_TEMP=${TEST_TEMP_LIST[$i]}
        SCORING_METHOD=${SCORING_LIST[0]}
        EPOCH=${EPOCH_LIST[0]}
        TEST_FREQUENCY=${TEST_LIST[0]}
        ACC_W=${ACC_W_LIST[$i]}
        Z3_W=${Z3_W_LIST[$i]}
        SWITCH_EPOCH=${SWITCH_EPOCH_LIST[$i]}
        SYSTEM_NAME=${SYSTEM_NAME_LIST[$i]}
        EVAL_PATH=${EVAL_PATH_LIST[$i]}
        DATA_PATH=${DATA_PATH_LIST[$i]}


        echo "Submitting job: TRAIN-TEMP=$TRAIN_TEMP, TEST-TEMP=$TEST_TEMP, SCORING-METHOD=$SCORING_METHOD, EPOCH=$EPOCH, TEST-FREQUENCY=$TEST_FREQUENCY,
        ACC_W=$ACC_W, Z3_W=$Z3_W, EPOCH_SWITCH=$SWITCH_EPOCH SYSTEM_NAME=$SYSTEM_NAME EVAL_PATH=$EVAL_PATH DATA_PATH=$DATA_PATH"
        sbatch $SLURM_SCRIPT_A100 $TRAIN_TEMP $TEST_TEMP $SCORING_METHOD $EPOCH $TEST_FREQUENCY $ACC_W $Z3_W $SWITCH_EPOCH $SYSTEM_NAME $EVAL_PATH $DATA_PATH
    done

    echo "All jobs submitted A100."
    echo "All scripts finished here."
fi






if false; then
    ACRONYM="STST"
    echo "Submitting Training job H200 + GT"
    TRAIN_TEMP_LIST=(0.8)
    TEST_TEMP_LIST=(0.8)
    SCORING_LIST=("gt")
    EPOCH_LIST=(1000)
    TEST_LIST=(5)
    ACC_W_LIST=(0.8)
    Z3_W_LIST=(0.2)
    SWITCH_EPOCH_LIST=(80)
    SYSTEM_NAME_LIST=("Reasoning360_sys_B_v3")
    EVAL_PATH_LIST=("small_train_small_test_1_parsed_v6a_${ACRONYM}")
    DATA_PATH_LIST=("ZebraPuzzle_to_guru_parsed_v6a_${ACRONYM}/small_train_small_test")



    SLURM_SCRIPT_H200="./scripts/train/Parsed_v2/${ACRONYM}_1_1shot_H200_zebra_rl_fsdp_parsed_v2.sh"
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
fi






if false; then
    ACRONYM="MTMT"
    echo "Submitting Training job H200 + GT"
    TRAIN_TEMP_LIST=(0.8)
    TEST_TEMP_LIST=(0.8)
    SCORING_LIST=("gt")
    EPOCH_LIST=(150)
    TEST_LIST=(5)
    ACC_W_LIST=(0.8)
    Z3_W_LIST=(0.2)
    SWITCH_EPOCH_LIST=(80)
    SYSTEM_NAME_LIST=("Reasoning360_sys_B_v3")
    EVAL_PATH_LIST=("small_train_small_test_1_parsed_v5d_${ACRONYM}")
    DATA_PATH_LIST=("ZebraPuzzle_to_guru_parsed_v5d_${ACRONYM}/med_train_med_test")



    SLURM_SCRIPT_H200="./scripts/train/Parsed_v2/${ACRONYM}_1_1shot_H200_zebra_rl_fsdp_parsed_v2.sh"
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

fi


























if false; then
    echo "Submitting Training job A100 + GT"
    TRAIN_TEMP_LIST=(0.8)
    TEST_TEMP_LIST=(0.8)
    SCORING_LIST=("gt")
    EPOCH_LIST=(150)
    TEST_LIST=(5)
    ACC_W_LIST=(0.8)
    Z3_W_LIST=(0.2)
    SWITCH_EPOCH_LIST=(80)
    SYSTEM_NAME_LIST=("Reasoning360_sys_B_v3")
    EVAL_PATH_LIST=("small_train_small_test_1_parsed_v5d_MTMT")
    DATA_PATH_LIST=("ZebraPuzzle_to_guru_parsed_v5d_MTMT/med_train_med_test")



    SLURM_SCRIPT_A100="./scripts/train/Parsed_v2/STST_1_1shot_A100_zebra_rl_fsdp_parsed_v2.sh"
    for i in "${!TRAIN_TEMP_LIST[@]}"; do
        TRAIN_TEMP=${TRAIN_TEMP_LIST[$i]}
        TEST_TEMP=${TEST_TEMP_LIST[$i]}
        SCORING_METHOD=${SCORING_LIST[0]}
        EPOCH=${EPOCH_LIST[0]}
        TEST_FREQUENCY=${TEST_LIST[0]}
        ACC_W=${ACC_W_LIST[$i]}
        Z3_W=${Z3_W_LIST[$i]}
        SWITCH_EPOCH=${SWITCH_EPOCH_LIST[$i]}
        SYSTEM_NAME=${SYSTEM_NAME_LIST[$i]}
        EVAL_PATH=${EVAL_PATH_LIST[$i]}
        DATA_PATH=${DATA_PATH_LIST[$i]}


        echo "Submitting job: TRAIN-TEMP=$TRAIN_TEMP, TEST-TEMP=$TEST_TEMP, SCORING-METHOD=$SCORING_METHOD, EPOCH=$EPOCH, TEST-FREQUENCY=$TEST_FREQUENCY,
        ACC_W=$ACC_W, Z3_W=$Z3_W, EPOCH_SWITCH=$SWITCH_EPOCH SYSTEM_NAME=$SYSTEM_NAME EVAL_PATH=$EVAL_PATH DATA_PATH=$DATA_PATH"
        sbatch $SLURM_SCRIPT_A100 $TRAIN_TEMP $TEST_TEMP $SCORING_METHOD $EPOCH $TEST_FREQUENCY $ACC_W $Z3_W $SWITCH_EPOCH $SYSTEM_NAME $EVAL_PATH $DATA_PATH
    done

    echo "All jobs submitted A100."
    echo "All scripts finished here."
fi




if false; then
    echo "Submitting Training job H200 + GT"
    TRAIN_TEMP_LIST=(0.8)
    TEST_TEMP_LIST=(0.8)
    SCORING_LIST=("gt")
    EPOCH_LIST=(150)
    TEST_LIST=(5)
    ACC_W_LIST=(0.8)
    Z3_W_LIST=(0.2)
    SWITCH_EPOCH_LIST=(80)
    SYSTEM_NAME_LIST=("Reasoning360_sys_B_v3")
    EVAL_PATH_LIST=("small_train_small_test_1_parsed_v5d")
    DATA_PATH_LIST=("ZebraPuzzle_to_guru_parsed_v5d/small_train_small_test")



    SLURM_SCRIPT_H200="./scripts/train/Parsed_v2/STST_1_1shot_H200_zebra_rl_fsdp_parsed_v2.sh"
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
fi










if false; then
    echo "Submitting Training job A100 + GT"
    TRAIN_TEMP_LIST=(0.8)
    TEST_TEMP_LIST=(0.8)
    SCORING_LIST=("gt")
    EPOCH_LIST=(150)
    TEST_LIST=(5)
    ACC_W_LIST=(0.8)
    Z3_W_LIST=(0.2)
    EPOCH_SWITCH_LIST=(50)


    SLURM_SCRIPT_A100="./scripts/train/Parsed_v2/STST_1_1shot_A100_zebra_rl_fsdp_parsed_v2.sh"
    for i in "${!TRAIN_TEMP_LIST[@]}"; do
        TRAIN_TEMP=${TRAIN_TEMP_LIST[$i]}
        TEST_TEMP=${TEST_TEMP_LIST[$i]}
        SCORING_METHOD=${SCORING_LIST[0]}
        EPOCH=${EPOCH_LIST[0]}
        TEST_FREQUENCY=${TEST_LIST[0]}
        ACC_W=${ACC_W_LIST[$i]}
        Z3_W=${Z3_W_LIST[$i]}
        EPOCH_SWITCH=${EPOCH_SWITCH_LIST[$i]}


        echo "Submitting job: TRAIN TEMPERATURE=$TEMP, TEST TEMPERATURE=, SCORING-METHOD=$SCORING_METHOD, EPOCH=$EPOCH, TEST-FREQUENCY=$TEST_FREQUENCY, ACC_W=$ACC_W, Z3_W=$Z3_W, EPOCH_SWITCH=$EPOCH_SWITCH"
        sbatch $SLURM_SCRIPT_A100 $TRAIN_TEMP $TEST_TEMP $SCORING_METHOD $EPOCH $TEST_FREQUENCY $ACC_W $Z3_W $EPOCH_SWITCH
    done

    echo "All jobs submitted A100."
    echo "All scripts finished here."
fi







if false; then
    echo "Submitting Training job A100"
    TEMP_LIST=(1.0)
    SCORING_LIST=("gt")
    EPOCH_LIST=(100)
    TEST_LIST=(5)
    SLURM_SCRIPT_A100="./scripts/train/STST_1_1shot_A100_zebra_rl_fsdp.sh"
    for i in "${!TEMP_LIST[@]}"; do
        TEMP=${TEMP_LIST[$i]}
        SCORING_METHOD=${SCORING_LIST[0]}
        EPOCH=${EPOCH_LIST[0]}
        TEST_FREQUENCY=${TEST_LIST[0]}

        echo "Submitting job: TEMPERATURE=$TEMP, SCORING-METHOD=$SCORING_METHOD EPOCH=$EPOCH TEST-FREQUENCY=$TEST_FREQUENCY"
        sbatch $SLURM_SCRIPT_A100 $TEMP $SCORING_METHOD $EPOCH $TEST_FREQUENCY
    done

    echo "All jobs submitted A100."
    echo "All scripts finished here."
fi



if false; then
    echo "Submitting Training job H200"
    TEMP_LIST=(0.8)
    SCORING_LIST=("gt+z3")
    EPOCH_LIST=(100)
    TEST_LIST=(5)
    SLURM_SCRIPT_H200="./scripts/train/STST_1_1shot_H200_zebra_rl_fsdp.sh"
    for i in "${!TEMP_LIST[@]}"; do
        TEMP=${TEMP_LIST[$i]}
        SCORING_METHOD=${SCORING_LIST[$i]}
        EPOCH=${EPOCH_LIST[$i]}
        TEST_FREQUENCY=${TEST_LIST[$i]}

        echo "Submitting job: TEMPERATURE=$TEMP, SCORING-METHOD=$SCORING_METHOD EPOCH=$EPOCH TEST-FREQUENCY=$TEST_FREQUENCY"
        sbatch $SLURM_SCRIPT_H200 $TEMP $SCORING_METHOD $EPOCH $TEST_FREQUENCY
    done

    echo "All jobs submitted H200."
fi










if false; then
    echo "Submitting Training job H200"
    DATA_CLASS_LIST=("correct_only")
    DATA_LIST=("/export/alt-insight/Reasoning360/Prompts_SFT_data/result_data_id_236791_time_20251216_081426_shortANS_jobid_237477"
              )
    EPOCH_LIST=(40)

    SLURM_SCRIPT_H200="./Prompts_System/run_prompts_H200.sh"


    for i in "${!DATA_CLASS_LIST[@]}"; do
        DATA_CLASS=${DATA_CLASS_LIST[$i]}
        DATA_PATH=${DATA_LIST[$i]}
        EPOCH=${EPOCH_LIST[$i]}

        echo "Submitting job: Data-Class=$DATA_CLASS Data_Path=$DATA_PATH Epoch=$EPOCH"
        sbatch $SLURM_SCRIPT_H200 $DATA_CLASS $DATA_PATH $EPOCH
    done


    echo "All jobs submitted H200."
    echo "All scripts finished here."
fi





if false; then
    echo "Submitting Evalution Job on A100"

    N_SAMPLES_LIST=(1)
    DATA_LIST=("medium")
    ATTEMPT_LIST=(2)
    MODE_LIST=("solution")
    LIMIT_LIST=(-1)
    REMOVE_CHKPT_LIST=(0)
    CHKPT_PATH_LIST=("/export/home/asifali/Reasoning360/Prompts_Checkpoint/data_job_id_237477_epoch_40_data_correct_only_20251216_142744_jobid_237674/checkpoint-2850"
                    "/export/home/asifali/Reasoning360/Prompts_Checkpoint/data_job_id_237477_epoch_40_data_correct_only_20251216_142744_jobid_237674/checkpoint-3000"
                    "/export/home/asifali/Reasoning360/Prompts_Checkpoint/data_job_id_237477_epoch_40_data_correct_only_20251216_142744_jobid_237674/checkpoint-3150"
                    "/export/home/asifali/Reasoning360/Prompts_Checkpoint/data_job_id_237477_epoch_40_data_correct_only_20251216_142744_jobid_237674/checkpoint-3300"
                    "/export/home/asifali/Reasoning360/Prompts_Checkpoint/data_job_id_237477_epoch_40_data_correct_only_20251216_142744_jobid_237674/checkpoint-3450"
                    "/export/home/asifali/Reasoning360/Prompts_Checkpoint/data_job_id_237477_epoch_40_data_correct_only_20251216_142744_jobid_237674/checkpoint-3600"
                    "/export/home/asifali/Reasoning360/Prompts_Checkpoint/data_job_id_237477_epoch_40_data_correct_only_20251216_142744_jobid_237674/checkpoint-3750"
                    "/export/home/asifali/Reasoning360/Prompts_Checkpoint/data_job_id_237477_epoch_40_data_correct_only_20251216_142744_jobid_237674/checkpoint-3900"
                    "/export/home/asifali/Reasoning360/Prompts_Checkpoint/data_job_id_237477_epoch_40_data_correct_only_20251216_142744_jobid_237674/checkpoint-4050"
                    )

    SLURM_SCRIPT_A100="./Prompts_System/run_prompts_A100.sh"

    for i in "${!CHKPT_PATH_LIST[@]}"; do
        N=${N_SAMPLES_LIST[0]}
        DATA=${DATA_LIST[0]}
        ATTEMPT=${ATTEMPT_LIST[0]}
        MODE=${MODE_LIST[0]}
        LIMIT=${LIMIT_LIST[0]}
        REMOVE_CHKPT=${REMOVE_CHKPT_LIST[0]}
        CHKPT_PATH=${CHKPT_PATH_LIST[$i]}
        echo "Submitting job: n_samples=$N data=$DATA attempt=$ATTEMPT mode=$MODE limit=$LIMIT remove $REMOVE_CHKPT path=$CHKPT_PATH"
        sbatch $SLURM_SCRIPT_A100 $N $DATA $ATTEMPT $MODE $LIMIT $REMOVE_CHKPT $CHKPT_PATH
    done

    echo "Finished All jobs submitted A100."
    echo " All scripts finished here...!"
fi





if false; then
  echo "Submitting Evalution Job on H200"

  N_SAMPLES_LIST=(1)
  DATA_LIST=("medium")
  ATTEMPT_LIST=(2)
  MODE_LIST=("solution")
  LIMIT_LIST=(-1)
  REMOVE_CHKPT_LIST=(0)
  CHKPT_PATH_LIST=("/export/home/asifali/Reasoning360/Prompts_Checkpoint/data_job_id_237477_epoch_40_data_correct_only_20251216_142744_jobid_237674/checkpoint-4200"
                "/export/home/asifali/Reasoning360/Prompts_Checkpoint/data_job_id_237477_epoch_40_data_correct_only_20251216_142744_jobid_237674/checkpoint-4350"
                "/export/home/asifali/Reasoning360/Prompts_Checkpoint/data_job_id_237477_epoch_40_data_correct_only_20251216_142744_jobid_237674/checkpoint-4500"
                "/export/home/asifali/Reasoning360/Prompts_Checkpoint/data_job_id_237477_epoch_40_data_correct_only_20251216_142744_jobid_237674/checkpoint-4650"
                "/export/home/asifali/Reasoning360/Prompts_Checkpoint/data_job_id_237477_epoch_40_data_correct_only_20251216_142744_jobid_237674/checkpoint-4800"
                "/export/home/asifali/Reasoning360/Prompts_Checkpoint/data_job_id_237477_epoch_40_data_correct_only_20251216_142744_jobid_237674/checkpoint-4950"
                "/export/home/asifali/Reasoning360/Prompts_Checkpoint/data_job_id_237477_epoch_40_data_correct_only_20251216_142744_jobid_237674/checkpoint-5100"
                "/export/home/asifali/Reasoning360/Prompts_Checkpoint/data_job_id_237477_epoch_40_data_correct_only_20251216_142744_jobid_237674/checkpoint-5250"
                "/export/home/asifali/Reasoning360/Prompts_Checkpoint/data_job_id_237477_epoch_40_data_correct_only_20251216_142744_jobid_237674/checkpoint-5400"
                "/export/home/asifali/Reasoning360/Prompts_Checkpoint/data_job_id_237477_epoch_40_data_correct_only_20251216_142744_jobid_237674/checkpoint-5550"
                "/export/home/asifali/Reasoning360/Prompts_Checkpoint/data_job_id_237477_epoch_40_data_correct_only_20251216_142744_jobid_237674/checkpoint-5700"
                "/export/home/asifali/Reasoning360/Prompts_Checkpoint/data_job_id_237477_epoch_40_data_correct_only_20251216_142744_jobid_237674/checkpoint-5850"
                "/export/home/asifali/Reasoning360/Prompts_Checkpoint/data_job_id_237477_epoch_40_data_correct_only_20251216_142744_jobid_237674/checkpoint-6000"
                )

  SLURM_SCRIPT_H200="./Prompts_System/run_prompts_H200.sh"

  for i in "${!CHKPT_PATH_LIST[@]}"; do
      N=${N_SAMPLES_LIST[0]}
      DATA=${DATA_LIST[0]}
      ATTEMPT=${ATTEMPT_LIST[0]}
      MODE=${MODE_LIST[0]}
      LIMIT=${LIMIT_LIST[0]}
      REMOVE_CHKPT=${REMOVE_CHKPT_LIST[0]}
      CHKPT_PATH=${CHKPT_PATH_LIST[$i]}
      echo "Submitting job: n_samples=$N data=$DATA attempt=$ATTEMPT mode=$MODE limit=$LIMIT remove $REMOVE_CHKPT path=$CHKPT_PATH"
      sbatch $SLURM_SCRIPT_H200 $N $DATA $ATTEMPT $MODE $LIMIT $REMOVE_CHKPT $CHKPT_PATH
  done

  echo "Finished All jobs submitted H200."
  echo " All scripts finished here...!"
fi


















if false; then

    echo "Submitting Evalution Job on A100"

    N_SAMPLES_LIST=(1)
    DATA_LIST=("small")
    ATTEMPT_LIST=(2)
    MODE_LIST=("solution")
    LIMIT_LIST=(-1)
    REMOVE_CHKPT_LIST=(true)
    CHKPT_PATH_LIST=("/export/home/asifali/Reasoning360/Prompts_Checkpoint/data_job_id_237477_epoch_8_data_correct_only_20251216_081953_jobid_237479/checkpoint-150"
                    "/export/home/asifali/Reasoning360/Prompts_Checkpoint/data_job_id_237477_epoch_8_data_correct_only_20251216_081953_jobid_237479/checkpoint-300"
                    "/export/home/asifali/Reasoning360/Prompts_Checkpoint/data_job_id_237477_epoch_8_data_correct_only_20251216_081953_jobid_237479/checkpoint-450"
                    "/export/home/asifali/Reasoning360/Prompts_Checkpoint/data_job_id_237477_epoch_8_data_correct_only_20251216_081953_jobid_237479/checkpoint-600"
                    "/export/home/asifali/Reasoning360/Prompts_Checkpoint/data_job_id_237477_epoch_8_data_correct_only_20251216_081953_jobid_237479/checkpoint-750"
                    "/export/home/asifali/Reasoning360/Prompts_Checkpoint/data_job_id_237477_epoch_8_data_correct_only_20251216_081953_jobid_237479/checkpoint-900"
                    "/export/home/asifali/Reasoning360/Prompts_Checkpoint/data_job_id_237477_epoch_8_data_correct_only_20251216_081953_jobid_237479/checkpoint-1050"
                    "/export/home/asifali/Reasoning360/Prompts_Checkpoint/data_job_id_237477_epoch_8_data_correct_only_20251216_081953_jobid_237479/checkpoint-1200"
                    )

    SLURM_SCRIPT_A100="./Prompts_System/run_prompts_A100.sh"

    for i in "${!CHKPT_PATH_LIST[@]}"; do
        N=${N_SAMPLES_LIST[0]}
        DATA=${DATA_LIST[0]}
        ATTEMPT=${ATTEMPT_LIST[0]}
        MODE=${MODE_LIST[0]}
        LIMIT=${LIMIT_LIST[0]}
        REMOVE_CHKPT=${REMOVE_CHKPT_LIST[0]}
        CHKPT_PATH=${CHKPT_PATH_LIST[$i]}
        echo "Submitting job: n_samples=$N data=$DATA attempt=$ATTEMPT mode=$MODE limit=$LIMIT remove $REMOVE_CHKPT path=$CHKPT_PATH"
        sbatch $SLURM_SCRIPT_A100 $N $DATA $ATTEMPT $MODE $LIMIT $REMOVE_CHKPT $CHKPT_PATH
    done

    echo "Finished All jobs submitted A100."
    echo " All scripts finished here...!"

fi











if false; then
    echo "Submitting Evaluation job H200"
    N_SAMPLES_LIST=(1)
    DATA_LIST=("small")
    ATTEMPT_LIST=(2)
    MODE_LIST=("solution")
    LIMIT_LIST=(-1)
    REMOVE_CHKPT_LIST=(true)
    CHKPT_PREFIX="/export/home/asifali/Reasoning360/Prompts_Checkpoint/data_job_id_237393_epoch_10_data_correct_only_20251215_222415_jobid_237445/checkpoint-69"

    SLURM_SCRIPT_H200="./Prompts_System/run_prompts_H200.sh"

    for i in {1..20}; do
        N=${N_SAMPLES_LIST[0]}
        DATA=${DATA_LIST[0]}
        ATTEMPT=${ATTEMPT_LIST[0]}
        MODE=${MODE_LIST[0]}
        LIMIT=${LIMIT_LIST[0]}
        REMOVE_CHKPT=${REMOVE_CHKPT_LIST[0]}
        CHKPT_PATH="${CHKPT_PREFIX}-$i"
        echo "Submitting job: n_samples=$N data=$DATA attempt=$ATTEMPT mode=$MODE limit=$LIMIT remove $REMOVE_CHKPT path=$CHKPT_PATH"
        sbatch $SLURM_SCRIPT_H200 $N $DATA $ATTEMPT $MODE $LIMIT $REMOVE_CHKPT $CHKPT_PATH
    done
    echo "All jobs submitted H200."
    echo "All scripts finished here...!"
fi
























if false; then
    echo "Submitting jobs CORRECT on H200"

    N_SAMPLES_LIST=(1)
    DATA_LIST=("small")
    ATTEMPT_LIST=(3)
    MODE_LIST=("solution")
    LIMIT_LIST=(-1)
    REMOVE_CHKPT_LIST=(true)
    CHKPT_PATH_LIST=("/export/home/asifali/Reasoning360/Prompts_Checkpoint/epoch_20_data_correct_only_20251215_090312_jobid_237245/")

    SLURM_SCRIPT_H200="./Prompts_System/run_prompts_H200.sh"

    for i in "${!N_SAMPLES_LIST[@]}"; do
        N=${N_SAMPLES_LIST[$i]}
        DATA=${DATA_LIST[$i]}
        ATTEMPT=${ATTEMPT_LIST[$i]}
        MODE=${MODE_LIST[$i]}
        LIMIT=${LIMIT_LIST[$i]}
        REMOVE_CHKPT=${REMOVE_CHKPT_LIST[$i]}
        CHKPT_PATH=${CHKPT_PATH_LIST[$i]}
        echo "Submitting job: n_samples=$N data=$DATA attempt=$ATTEMPT mode=$MODE limit=$LIMIT remove $REMOVE_CHKPT path=$CHKPT_PATH"
        sbatch $SLURM_SCRIPT_H200 $N $DATA $ATTEMPT $MODE $LIMIT $REMOVE_CHKPT $CHKPT_PATH
    done

    echo "Finished All jobs submitted H200."
    echo " All scripts finished here...!"

fi



if false; then



if false; then
    echo "Submitting jobs on H200"

    DATA_CLASS_LIST=("correct_only")
    DATA_LIST=("/export/home/asifali/Reasoning360/Prompts_SFT_data/previous_id_236791_time_20251214_181753_shortANS_jobid_237144/")
    EPOCH_LIST=(20)
    EVAL_LIST=(82)
    SAVE_LIST=(164)

    SLURM_SCRIPT_H200="./Prompts_System/run_prompts_H200.sh"


    for i in "${!DATA_CLASS_LIST[@]}"; do
        DATA_CLASS=${DATA_CLASS_LIST[$i]}
        DATA_PATH=${DATA_LIST[$i]}
        EPOCH=${EPOCH_LIST[$i]}
        EVAL_STEP=${EVAL_LIST[$i]}
        SAVE_STEP=${SAVE_LIST[$i]}
        echo "Submitting job: Data-Class=$DATA_CLASS Data_Path=$DATA_PATH Epoch=$EPOCH Eval-Step=$EVAL_STEP Save-Step=$SAVE_STEP"
        sbatch $SLURM_SCRIPT_H200 $DATA_CLASS $DATA_PATH $EPOCH $EVAL_STEP $SAVE_STEP
    done

    echo "All jobs submitted H200."
    echo " All scripts finished here...!"




    echo "Submitting jobs COMBINED on A100"

    N_SAMPLES_LIST=(1 1 1 1 1)
    DATA_LIST=("small" "small" "small" "small" "small")
    ATTEMPT_LIST=(3 3 3 3 3)
    MODE_LIST=("solution" "solution" "solution" "solution" "solution")

    CHKPT_PATH_LIST=("/export/home/asifali/Reasoning360/Prompts_Checkpoint/epoch_20_data_combined_20251214_202205_jobid_237173/checkpoint-2568"
                    "/export/home/asifali/Reasoning360/Prompts_Checkpoint/epoch_20_data_combined_20251214_202205_jobid_237173/checkpoint-2996"
                    "/export/home/asifali/Reasoning360/Prompts_Checkpoint/epoch_20_data_combined_20251214_202205_jobid_237173/checkpoint-3424"
                    "/export/home/asifali/Reasoning360/Prompts_Checkpoint/epoch_20_data_combined_20251214_202205_jobid_237173/checkpoint-3852"
                    "/export/home/asifali/Reasoning360/Prompts_Checkpoint/epoch_20_data_combined_20251214_202205_jobid_237173/checkpoint-4280"
                    )

    SLURM_SCRIPT_A100="./Prompts_System/run_prompts_A100.sh"

    for i in "${!N_SAMPLES_LIST[@]}"; do
        N=${N_SAMPLES_LIST[$i]}
        DATA=${DATA_LIST[$i]}
        ATTEMPT=${ATTEMPT_LIST[$i]}
        MODE=${MODE_LIST[$i]}
        CHKPT_PATH=${CHKPT_PATH_LIST[$i]}
        echo "Submitting job: n_samples=$N data=$DATA attempt=$ATTEMPT mode=$MODE path=$CHKPT_PATH"
        sbatch $SLURM_SCRIPT_A100 $N $DATA $ATTEMPT $MODE $CHKPT_PATH
    done

    echo "Finished All jobs submitted A100."
    echo " All scripts finished here...!"






    DATA_CLASS_LIST=("combined")
    DATA_LIST=("/export/home/asifali/Reasoning360/Prompts_SFT_data/previous_id_236791_time_20251214_181753_shortANS_jobid_237144/")
    EPOCH_LIST=(20)
    EVAL_LIST=(214)
    SAVE_LIST=(428)

    SLURM_SCRIPT_A100="./Prompts_System/run_prompts_A100.sh"


    echo "Submitting jobs..."

    for i in "${!DATA_CLASS_LIST[@]}"; do
        DATA_CLASS=${DATA_CLASS_LIST[$i]}
        DATA_PATH=${DATA_LIST[$i]}
        EPOCH=${EPOCH_LIST[$i]}
        EVAL_STEP=${EVAL_LIST[$i]}
        SAVE_STEP=${SAVE_LIST[$i]}
        echo "Submitting job: Data-Class=$DATA_CLASS Data_Path=$DATA_PATH Epoch=$EPOCH Eval-Step=$EVAL_STEP Save-Step=$SAVE_STEP"
        sbatch $SLURM_SCRIPT_A100 $DATA_CLASS $DATA_PATH $EPOCH $EVAL_STEP $SAVE_STEP
    done

    echo "All jobs submitted A100."
    echo " All scripts finished here...!"





    echo "Submitting jobs COMBINED on H200"

    N_SAMPLES_LIST=(1 1 1 1 1 1 1 1 1 1)
    DATA_LIST=("small" "small" "small" "small" "small" "small" "small" "small" "small" "small")
    ATTEMPT_LIST=(3 3 3 3 3 3 3 3 3 3)
    MODE_LIST=("solution" "solution" "solution" "solution" "solution" "solution" "solution" "solution" "solution" "solution")

    CHKPT_PATH_LIST=("/export/home/asifali/Reasoning360/Prompts_Checkpoint/epoch_10_data_combined_20251214_180032_jobid_237142/checkpoint-180"
                     "/export/home/asifali/Reasoning360/Prompts_Checkpoint/epoch_10_data_combined_20251214_180032_jobid_237142/checkpoint-360"
                     "/export/home/asifali/Reasoning360/Prompts_Checkpoint/epoch_10_data_combined_20251214_180032_jobid_237142/checkpoint-540"
                     "/export/home/asifali/Reasoning360/Prompts_Checkpoint/epoch_10_data_combined_20251214_180032_jobid_237142/checkpoint-720"
                     "/export/home/asifali/Reasoning360/Prompts_Checkpoint/epoch_10_data_combined_20251214_180032_jobid_237142/checkpoint-900"
                     "/export/home/asifali/Reasoning360/Prompts_Checkpoint/epoch_10_data_combined_20251214_180032_jobid_237142/checkpoint-1080"
                     "/export/home/asifali/Reasoning360/Prompts_Checkpoint/epoch_10_data_combined_20251214_180032_jobid_237142/checkpoint-1260"
                     "/export/home/asifali/Reasoning360/Prompts_Checkpoint/epoch_10_data_combined_20251214_180032_jobid_237142/checkpoint-1440"
                     "/export/home/asifali/Reasoning360/Prompts_Checkpoint/epoch_10_data_combined_20251214_180032_jobid_237142/checkpoint-1620"
                     "/export/home/asifali/Reasoning360/Prompts_Checkpoint/epoch_10_data_combined_20251214_180032_jobid_237142/checkpoint-1800"
                    )

    SLURM_SCRIPT_H200="./Prompts_System/run_prompts_H200.sh"

    for i in "${!N_SAMPLES_LIST[@]}"; do
        N=${N_SAMPLES_LIST[$i]}
        DATA=${DATA_LIST[$i]}
        ATTEMPT=${ATTEMPT_LIST[$i]}
        MODE=${MODE_LIST[$i]}
        CHKPT_PATH=${CHKPT_PATH_LIST[$i]}
        echo "Submitting job: n_samples=$N data=$DATA attempt=$ATTEMPT mode=$MODE path=$CHKPT_PATH"
        sbatch $SLURM_SCRIPT_H200 $N $DATA $ATTEMPT $MODE $CHKPT_PATH
    done

    echo "Finished All jobs submitted H200."
    echo " All scripts finished here...!"



    echo "Submitting jobs CORRECT on A100"
    N_SAMPLES_LIST=(1 1 1 1 1)
    DATA_LIST=("small" "small" "small" "small" "small")
    ATTEMPT_LIST=(2 2 2 2 2)
    MODE_LIST=("solution" "solution" "solution" "solution" "solution")

    CHKPT_PATH_LIST=("/export/home/asifali/Reasoning360/Prompts_Checkpoint/epoch_5_data_correct_only_20251214_180032_jobid_237143/checkpoint-70"
                    "/export/home/asifali/Reasoning360/Prompts_Checkpoint/epoch_5_data_correct_only_20251214_180032_jobid_237143/checkpoint-140"
                    "/export/home/asifali/Reasoning360/Prompts_Checkpoint/epoch_5_data_correct_only_20251214_180032_jobid_237143/checkpoint-210"
                    "/export/home/asifali/Reasoning360/Prompts_Checkpoint/epoch_5_data_correct_only_20251214_180032_jobid_237143/checkpoint-280"
                    "/export/home/asifali/Reasoning360/Prompts_Checkpoint/epoch_5_data_correct_only_20251214_180032_jobid_237143/checkpoint-345"
                    )

    SLURM_SCRIPT_A100="./Prompts_System/run_prompts_A100.sh"

    for i in "${!N_SAMPLES_LIST[@]}"; do
        N=${N_SAMPLES_LIST[$i]}
        DATA=${DATA_LIST[$i]}
        ATTEMPT=${ATTEMPT_LIST[$i]}
        MODE=${MODE_LIST[$i]}
        CHKPT_PATH=${CHKPT_PATH_LIST[$i]}
        echo "Submitting job: n_samples=$N data=$DATA attempt=$ATTEMPT mode=$MODE path=$CHKPT_PATH"
        sbatch $SLURM_SCRIPT_A100 $N $DATA $ATTEMPT $MODE $CHKPT_PATH
    done

    echo "Finished All jobs submitted A100."
fi

if false; then
    echo "Submitting jobs COMBINED on H200"

    N_SAMPLES_LIST=(1 1 1 1 1 1 1 1 1 1)
    DATA_LIST=("small" "small" "small" "small" "small" "small" "small" "small" "small" "small")
    ATTEMPT_LIST=(3 3 3 3 3 3 3 3 3 3)
    MODE_LIST=("solution" "solution" "solution" "solution" "solution" "solution" "solution" "solution" "solution" "solution")

    CHKPT_PATH_LIST=("/export/home/asifali/Reasoning360/Prompts_Checkpoint/epoch_10_data_combined_20251214_163646_jobid_237138/checkpoint-360"
                    "/export/home/asifali/Reasoning360/Prompts_Checkpoint/epoch_10_data_combined_20251214_163646_jobid_237138/checkpoint-540"
                    )

    SLURM_SCRIPT_H200="./Prompts_System/run_prompts_H200.sh"

    for i in "${!N_SAMPLES_LIST[@]}"; do
        N=${N_SAMPLES_LIST[$i]}
        DATA=${DATA_LIST[$i]}
        ATTEMPT=${ATTEMPT_LIST[$i]}
        MODE=${MODE_LIST[$i]}
        CHKPT_PATH=${CHKPT_PATH_LIST[$i]}
        echo "Submitting job: n_samples=$N data=$DATA attempt=$ATTEMPT mode=$MODE path=$CHKPT_PATH"
        sbatch $SLURM_SCRIPT_H200 $N $DATA $ATTEMPT $MODE $CHKPT_PATH
    done

    echo "Finished All jobs submitted H200."
    echo " All scripts finished here...!"
fi






if false; then
    DATA_CLASS_LIST=("combined" "correct_only")
    DATA_LIST=("/export/home/asifali/Reasoning360/Prompts_SFT_data/previous_id_236791_time_20251214_141512_jobid_237101/"
               "/export/home/asifali/Reasoning360/Prompts_SFT_data/previous_id_236791_time_20251214_141512_jobid_237101/")
    EPOCH_LIST=(10 5)
    EVAL_LIST=(180 70)
    SAVE_LIST=(180 70)

    SLURM_SCRIPT_H200="./Prompts_System/run_prompts_H200.sh"


    echo "Submitting jobs..."

    for i in "${!DATA_CLASS_LIST[@]}"; do
        DATA_CLASS=${DATA_CLASS_LIST[$i]}
        DATA_PATH=${DATA_LIST[$i]}
        EPOCH=${EPOCH_LIST[$i]}
        EVAL_STEP=${EVAL_LIST[$i]}
        SAVE_STEP=${SAVE_LIST[$i]}
        echo "Submitting job: Data-Class=$DATA_CLASS Data_Path=$DATA_PATH Epoch=$EPOCH Eval-Step=$EVAL_STEP Save-Step=$SAVE_STEP"
        sbatch $SLURM_SCRIPT_H200 $DATA_CLASS $DATA_PATH $EPOCH $EVAL_STEP $SAVE_STEP
    done

    echo "All jobs submitted H200."
    echo " All scripts finished here...!"




    N_SAMPLES_LIST=(1 1)
    DATA_LIST=("small" "small")
    ATTEMPT_LIST=(3 3)
    MODE_LIST=("solution" "solution")

    CHKPT_PATH_LIST=("/export/home/asifali/Reasoning360/Prompts_Checkpoint/epoch_10_data_combined_20251214_163646_jobid_237138/checkpoint-360"
                    "/export/home/asifali/Reasoning360/Prompts_Checkpoint/epoch_10_data_combined_20251214_163646_jobid_237138/checkpoint-540"
                    )

    SLURM_SCRIPT_A100="./Prompts_System/run_prompts_A100.sh"


    echo "Submitting jobs..."

    for i in "${!N_SAMPLES_LIST[@]}"; do
        N=${N_SAMPLES_LIST[$i]}
        DATA=${DATA_LIST[$i]}
        ATTEMPT=${ATTEMPT_LIST[$i]}
        MODE=${MODE_LIST[$i]}
        CHKPT_PATH=${CHKPT_PATH_LIST[$i]}
        echo "Submitting job: n_samples=$N data=$DATA attempt=$ATTEMPT mode=$MODE path=$CHKPT_PATH"
        sbatch $SLURM_SCRIPT_A100 $N $DATA $ATTEMPT $MODE $CHKPT_PATH
    done

    echo "All jobs submitted A100."
    echo " All scripts finished here...!"





    DATA_CLASS_LIST=("combined" "correct_only")
    DATA_LIST=("/export/home/asifali/Reasoning360/Prompts_SFT_data/previous_id_236791_time_20251214_141512_jobid_237101/"
               "/export/home/asifali/Reasoning360/Prompts_SFT_data/previous_id_236791_time_20251214_141512_jobid_237101/")
    EPOCH_LIST=(10 5)
    EVAL_LIST=(180 70)
    SAVE_LIST=(180 70)

    SLURM_SCRIPT_H200="./Prompts_System/run_prompts_H200.sh"


    echo "Submitting jobs..."

    for i in "${!DATA_CLASS_LIST[@]}"; do
        DATA_CLASS=${DATA_CLASS_LIST[$i]}
        DATA_PATH=${DATA_LIST[$i]}
        EPOCH=${EPOCH_LIST[$i]}
        EVAL_STEP=${EVAL_LIST[$i]}
        SAVE_STEP=${SAVE_LIST[$i]}
        echo "Submitting job: Data=$DATA Epoch=$EPOCH Eval-Step=$EVAL_STEP Save-Step=$SAVE_STEP"
        sbatch $SLURM_SCRIPT_H200 $DATA_CLASS $DATA_PATH $EPOCH $EVAL_STEP $SAVE_STEP
    done

    echo "All jobs submitted H200."
    echo " All scripts finished here...!"





    N_SAMPLES_LIST=(1)
    DATA_LIST=("small")
    ATTEMPT_LIST=(3)
    MODE_LIST=("solution")
    CHKPT_PATH_LIST=("/export/home/asifali/Reasoning360/Prompts_Checkpoint/data_combined_20251213_153420_jobid_236881/checkpoint-500")

    SLURM_SCRIPT_A100="./Prompts_System/run_prompts_A100.sh"


    echo "Submitting jobs..."

    for i in "${!N_SAMPLES_LIST[@]}"; do
        N=${N_SAMPLES_LIST[$i]}
        DATA=${DATA_LIST[$i]}
        ATTEMPT=${ATTEMPT_LIST[$i]}
        MODE=${MODE_LIST[$i]}
        CHKPT_PATH=${CHKPT_PATH_LIST[$i]}
        echo "Submitting job: n_samples=$N data=$DATA attempt=$ATTEMPT mode=$MODE path=$CHKPT_PATH"
        sbatch $SLURM_SCRIPT_A100 $N $DATA $ATTEMPT $MODE $CHKPT_PATH
    done

    echo "All jobs submitted A-100."
    echo " All scripts finished here...!"






    #echo "All jobs submitted A-100."
    #echo "script finished"


    N_SAMPLES_LIST=(1 1 1 1)
    DATA_LIST=("small" "small" "small" "small")
    ATTEMPT_LIST=(5 5 5 5)
    MODE_LIST=("solution" "solution" "solution" "solution")
    CHKPT_PATH_LIST=("/export/home/asifali/Reasoning360/Prompts_Checkpoint/data_combined_20251212_185031_jobid_236828"
                     "/export/home/asifali/Reasoning360/Prompts_Checkpoint/data_combined_20251212_185031_jobid_236828/checkpoint-100"
                     "/export/home/asifali/Reasoning360/Prompts_Checkpoint/data_combined_20251212_185031_jobid_236828/checkpoint-180"
                     "/export/home/asifali/Reasoning360/Prompts_Checkpoint/epoch_3_data_correct_only_20251213_164324_jobid_236890")

    SLURM_SCRIPT_H200="./Prompts_System/run_prompts_H200.sh"


    echo "Submitting jobs..."

    for i in "${!N_SAMPLES_LIST[@]}"; do
        N=${N_SAMPLES_LIST[$i]}
        DATA=${DATA_LIST[$i]}
        ATTEMPT=${ATTEMPT_LIST[$i]}
        MODE=${MODE_LIST[$i]}
        CHKPT_PATH=${CHKPT_PATH_LIST[$i]}
        echo "Submitting job: n_samples=$N data=$DATA attempt=$ATTEMPT mode=$MODE path=$CHKPT_PATH"
        sbatch $SLURM_SCRIPT_H200 $N $DATA $ATTEMPT $MODE $CHKPT_PATH
    done

    echo "All jobs submitted H-200."
    echo " All scripts finished here...!"





    N_SAMPLES_LIST=(1 1)
    DATA_LIST=("small" "small" )
    ATTEMPT_LIST=(10 10)
    MODE_LIST=("solution" "solution")
    CHKPT_PATH_LIST=("/export/home/asifali/Reasoning360/Prompts_Checkpoint/epoch_2_data_combined_20251213_164325_jobid_236889"
                     "/export/home/asifali/Reasoning360/Prompts_Checkpoint/epoch_3_data_correct_only_20251213_164324_jobid_236890"
                     )

    SLURM_SCRIPT_A100="./Prompts_System/run_prompts_A100.sh"


    echo "Submitting jobs..."

    for i in "${!N_SAMPLES_LIST[@]}"; do
        N=${N_SAMPLES_LIST[$i]}
        DATA=${DATA_LIST[$i]}
        ATTEMPT=${ATTEMPT_LIST[$i]}
        MODE=${MODE_LIST[$i]}
        CHKPT_PATH=${CHKPT_PATH_LIST[$i]}
        echo "Submitting job: n_samples=$N data=$DATA attempt=$ATTEMPT mode=$MODE path=$CHKPT_PATH"
        sbatch $SLURM_SCRIPT_A100 $N $DATA $ATTEMPT $MODE $CHKPT_PATH
    done

    echo "All jobs submitted A-100...!"
    echo "All scripts finished here...!"






    DATA_LIST=("combined" "correct_only")
    EPOCH_LIST=(2 3)
    EVAL_LIST=(100 100)
    SAVE_LIST=(500 500)

    SLURM_SCRIPT_A100="./Prompts_System/run_prompts_A100.sh"


    echo "Submitting jobs..."

    for i in "${!DATA_LIST[@]}"; do
        DATA=${DATA_LIST[$i]}
        EPOCH=${EPOCH_LIST[$i]}
        EVAL_STEP=${EVAL_LIST[$i]}
        SAVE_STEP=${SAVE_LIST[$i]}
        echo "Submitting job: Data=$DATA Epoch=$EPOCH Eval-Step=$EVAL_STEP Save-Step=$SAVE_STEP"
        sbatch $SLURM_SCRIPT_A100 $DATA $EPOCH $EVAL_STEP $SAVE_STEP
    done

    echo "All jobs submitted A100."
    echo " All scripts finished here...!"






    N_SAMPLES_LIST=(1 1 1 1)
    DATA_LIST=("small" "small" "small" "small")
    ATTEMPT_LIST=(10 10 10 10)
    MODE_LIST=("solution" "solution" "solution" "solution")
    CHKPT_PATH_LIST=("/export/home/asifali/Reasoning360/Prompts_Checkpoint/data_correct_only_20251213_151232_jobid_236878/checkpoint-294"
                     "/export/home/asifali/Reasoning360/Prompts_Checkpoint/data_correct_only_20251213_151234_jobid_236879/checkpoint-294"
                     "/export/home/asifali/Reasoning360/Prompts_Checkpoint/data_combined_20251213_151218_jobid_236880/checkpoint-500"
                     "/export/home/asifali/Reasoning360/Prompts_Checkpoint/data_combined_20251213_153420_jobid_236881/checkpoint-500")

    SLURM_SCRIPT_H200="./Prompts_System/run_prompts_H200.sh"


    echo "Submitting jobs..."

    for i in "${!N_SAMPLES_LIST[@]}"; do
        N=${N_SAMPLES_LIST[$i]}
        DATA=${DATA_LIST[$i]}
        ATTEMPT=${ATTEMPT_LIST[$i]}
        MODE=${MODE_LIST[$i]}
        CHKPT_PATH=${CHKPT_PATH_LIST[$i]}
        echo "Submitting job: n_samples=$N data=$DATA attempt=$ATTEMPT mode=$MODE path=$CHKPT_PATH"
        sbatch $SLURM_SCRIPT_H200 $N $DATA $ATTEMPT $MODE $CHKPT_PATH
    done

    echo "All jobs submitted H-200."
    echo " All scripts finished here...!"






    DATA_LIST=("correct_only" "correct_only" "combined" "combined")
    EPOCH_LIST=(2 2 3 3)
    EVAL_LIST=(100 100 100 100)
    SAVE_LIST=(500 500 500 500)

    SLURM_SCRIPT_A100="./Prompts_System/run_prompts_A100.sh"


    echo "Submitting jobs..."

    for i in "${!DATA_LIST[@]}"; do
        DATA=${DATA_LIST[$i]}
        EPOCH=${EPOCH_LIST[$i]}
        EVAL_STEP=${EVAL_LIST[$i]}
        SAVE_STEP=${SAVE_LIST[$i]}
        echo "Submitting job: Data=$DATA Epoch=$EPOCH Eval-Step=$EVAL_STEP Save-Step=$SAVE_STEP"
        sbatch $SLURM_SCRIPT_A100 $DATA $EPOCH $EVAL_STEP $SAVE_STEP
    done

    echo "All jobs submitted A100."
    echo " All scripts finished here...!"











    N_SAMPLES_LIST=(1 1 1)
    DATA_LIST=("small" "small" "small")
    ATTEMPT_LIST=(10 10 10)
    MODE_LIST=("solution" "solution" "solution")
    CHKPT_PATH_LIST=("/export/home/asifali/Reasoning360/Prompts_Checkpoint/data_combined_20251212_185031_jobid_236828/checkpoint-100"
                     "/export/home/asifali/Reasoning360/Prompts_Checkpoint/data_combined_20251212_185031_jobid_236828/checkpoint-180"
                     "/export/home/asifali/Reasoning360/Prompts_Checkpoint/data_combined_20251212_185031_jobid_236828")

    SLURM_SCRIPT_H200="./Prompts_System/run_prompts_H200.sh"


    echo "Submitting jobs..."

    for i in "${!N_SAMPLES_LIST[@]}"; do
        N=${N_SAMPLES_LIST[$i]}
        DATA=${DATA_LIST[$i]}
        ATTEMPT=${ATTEMPT_LIST[$i]}
        MODE=${MODE_LIST[$i]}
        CHKPT_PATH=${CHKPT_PATH_LIST[$i]}
        echo "Submitting job: n_samples=$N data=$DATA attempt=$ATTEMPT mode=$MODE path=$CHKPT_PATH"
        sbatch $SLURM_SCRIPT_H200 $N $DATA $ATTEMPT $MODE $CHKPT_PATH
    done

    echo "All jobs submitted H-200."
    echo " All scripts finished here...!"




    N_SAMPLES_LIST=(1 2 3 4 9 10)
    MODE_LIST=("solution" "solution" "solution" "solution" "solution" "solution")

    SLURM_SCRIPT_A100="./Prompts_System/run_prompts_A100.sh"


    echo "Submitting jobs..."

    for i in "${!N_SAMPLES_LIST[@]}"; do
        N=${N_SAMPLES_LIST[$i]}
        MODE=${MODE_LIST[$i]}

        echo "Submitting job: n_samples=$N mode=$MODE"
        sbatch $SLURM_SCRIPT_A100 $N $MODE
    done

    echo "All jobs submitted A100."



    N_SAMPLES_LIST=(1)
    DATA_LIST=("small")
    ATTEMPT_LIST=(10)
    MODE_LIST=("solution")

    SLURM_SCRIPT_A100="./Prompts_System/run_prompts_A100.sh"


    echo "Submitting jobs..."

    for i in "${!N_SAMPLES_LIST[@]}"; do
        N=${N_SAMPLES_LIST[$i]}
        DATA=${DATA_LIST[$i]}
        ATTEMPT=${ATTEMPT_LIST[$i]}
        MODE=${MODE_LIST[$i]}

        echo "Submitting job: n_samples=$N data=$DATA attempt=$ATTEMPT mode=$MODE"
        sbatch $SLURM_SCRIPT_A100 $N $DATA $ATTEMPT $MODE
    done

    echo "All jobs submitted A-100."
    echo "script finished"

fi


if false; then
    N_SAMPLES_LIST=(1 2 3)
    DATA_LIST=("small" "small" "small")
    ATTEMPT_LIST=(10 10 10)
    MODE_LIST=("solution" "solution" "solution")

    SLURM_SCRIPT_H200="./Prompts_System/run_prompts_H200.sh"


    echo "Submitting jobs..."

    for i in "${!N_SAMPLES_LIST[@]}"; do
        N=${N_SAMPLES_LIST[$i]}
        DATA=${DATA_LIST[$i]}
        ATTEMPT=${ATTEMPT_LIST[$i]}
        MODE=${MODE_LIST[$i]}

        echo "Submitting job: n_samples=$N data=$DATA attempt=$ATTEMPT mode=$MODE"
        sbatch $SLURM_SCRIPT_H200 $N $DATA $ATTEMPT $MODE
    done

    echo "All jobs submitted H-200."
    echo "script finished"




    # Values of n_samples you want to run
    N_SAMPLES_LIST=(1 2 3)
    DATA_LIST=("small" "small" "small")
    ATTEMPT_LIST=(10 10 10)
    MODE_LIST=("solution" "solution" "solution")

    SLURM_SCRIPT_H200="./Prompts_System/run_prompts_H200.sh"


    echo "Submitting jobs..."

    for i in "${!N_SAMPLES_LIST[@]}"; do
        N=${N_SAMPLES_LIST[$i]}
        DATA=${DATA_LIST[$i]}
        ATTEMPT=${ATTEMPT_LIST[$i]}
        MODE=${MODE_LIST[$i]}

        echo "Submitting job: n_samples=$N data=$DATA attempt=$ATTEMPT mode=$MODE"
        sbatch $SLURM_SCRIPT_H200 $N $DATA $ATTEMPT $MODE
    done

    echo "All jobs submitted H-200."
    echo "script finished"
fi



# Values of n_samples you want to run
if false; then
    N_SAMPLES_LIST=(8 9)
    MODE_LIST=("solution" "solution")

    SLURM_SCRIPT_A100="./Prompts_System/run_prompts_A100.sh"


    echo "Submitting jobs..."

    for i in "${!N_SAMPLES_LIST[@]}"; do
        N=${N_SAMPLES_LIST[$i]}
        MODE=${MODE_LIST[$i]}

        echo "Submitting job: n_samples=$N mode=$MODE"
        sbatch $SLURM_SCRIPT_A100 $N $MODE
    done

    echo "All jobs submitted A100."
fi


if false; then
    # Values of n_samples you want to run
    N_SAMPLES_LIST=(1 2 3 4)
    MODE_LIST=("solution" "solution" "solution" "solution")

    SLURM_SCRIPT_A100="./Prompts_System/run_prompts_A100.sh"


    echo "Submitting jobs..."

    for i in "${!N_SAMPLES_LIST[@]}"; do
        N=${N_SAMPLES_LIST[$i]}
        MODE=${MODE_LIST[$i]}

        echo "Submitting job: n_samples=$N mode=$MODE"
        sbatch $SLURM_SCRIPT_A100 $N $MODE
    done

    echo "All jobs submitted A100."




    # Values of n_samples you want to run
    N_SAMPLES_LIST=(5 7 10)
    MODE_LIST=("solution" "solution" "solution")

    SLURM_SCRIPT_H200="./Prompts_System/run_prompts_H200.sh"


    echo "Submitting jobs..."

    for i in "${!N_SAMPLES_LIST[@]}"; do
        N=${N_SAMPLES_LIST[$i]}
        MODE=${MODE_LIST[$i]}

        echo "Submitting job: n_samples=$N mode=$MODE"
        sbatch $SLURM_SCRIPT_H200 $N $MODE
    done

    echo "All jobs submitted."
fi

fi

