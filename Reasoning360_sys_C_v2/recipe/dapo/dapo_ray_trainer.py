# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import uuid
from collections import defaultdict
from copy import deepcopy
from pprint import pprint
import os
import json
import time
import datasets
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    RayPPOTrainer,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.utils.profiler import marked_timer




job_id = os.getenv("SLURM_JOB_ID")




def save_dataset(dataset, output_dir, filename, sample_size=None):
    """
    Save a dataset to a parquet file with appropriate naming.

    Args:
        dataset: The dataset to save
        output_dir: Directory to save the dataset
        filename_prefix: Base filename to use
        sample_size: Sample size to add as suffix to filename

    Returns:
        str: Path to the saved file
    """
    # Add suffix based on actual dataset size if sample_size is None
    if sample_size is None:
        sample_size = len(dataset)

    # Create filename with appropriate suffix
    output_path = os.path.join(output_dir, f"{filename}.parquet")

    # Save dataset
    dataset.to_parquet(output_path)

    return output_path

def extract_clues_from_puzzle(puzzle_text):
    """Extract clues from the puzzle text."""
    if "## Clues:" in puzzle_text:
        clues_part = puzzle_text.split("## Clues:")[1]
        # Extract each clue line
        clues = []
        for line in clues_part.splitlines():
            line = line.strip()
            if line and line[0].isdigit() and "." in line:
                # Remove the numbering and keep the clue text
                clue_text = line.split(".", 1)[1].strip()
                clues.append(clue_text)
        return clues
    else:
        return []


def make_map_fn(split, data_source):
    def process_fn(example, idx):
        # Use 'ground_truth' instead of 'solution' since that's what the input data has
        final_grid = example['solution']

        # Use the 'clues' field directly from the input data
        clues = extract_clues_from_puzzle(puzzle_text=example['puzzle'])

        #system_prompt = SOLUTION_PROMPT_SYSTEM_SOLUTION_BASED
        # user_prompt = SOLUTION_PROMPT_USER_SOLUTION_BASED.format(puzzle=example['puzzle'])
        user_prompt = example['prompt']

        data = {
            "data_source": data_source,
            "prompt": [
                {
                    "role": "user",
                    "content": user_prompt
                }],
            'raw_prompt': [
                {
                    "role": "user",
                    "content": user_prompt
                }],
            "ability": "logical_reasoning",
            "reward_model": {
                "style": "rule",
                "ground_truth": final_grid,
            },
            "apply_chat_template": False,
            "extra_info": {
                'id': example['id'] if 'id' in example else str(idx),
                'split': split,
                'clues': clues
            }
        }

        #if idx == 0:
        #    print(f"data_source: {data_source}, split: {split}, idx: {idx}")
        #    print("\n" + "=" * 100 + f"{data_source} {split} {idx}" + "=" * 10)
        #    print(data)
        #    print("\n\n")
        return data

    return process_fn






class RayDAPOTrainer(RayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        if os.environ.get("DEBUG_CODE", "0").lower() in ("1", "true", "yes"):
            print(f"DEBUG-MODE: Loading Checkpoint:")
        self._load_checkpoint()


        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            os.environ["STEP1_STATUS"] = "1"
            os.environ["CURRENT_EPOCH"] = str(0)
            print(f"VALIDATE BEFORE MODEL TRAINING")
            os.environ["VALID_STATUS"] = "1"
            val_metrics = self._validate()
            os.environ["VALID_STATUS"] = "2"
            val_metrics_tr = self._validate_tr()
            os.environ["VALID_STATUS"] = "0"
            assert val_metrics, f"{val_metrics=}"
            pprint(f"INITIAL VALIDATION METRICS: {val_metrics}")
            print()
            pprint(f"INITIAL TR-VALIDATION METRICS: {val_metrics_tr}")
            print()
            ##logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        timing_raw = defaultdict(float)

        # Step-2 configuration
        step2_iterations = self.config.trainer.get("step2_iterations", 1)
        enable_step2 = self.config.trainer.get("enable_step2", True)
        write_step_outputs = self.config.trainer.get("write_step_outputs", True)
        step2_mini_batch_iteration = self.config.trainer.get("step2_mini_batch_iteration", 1)

        STEP1_SCORINING = self.config.trainer.get("STEP1_SCORINING", True)
        STEP2_SCORINING = self.config.trainer.get("STEP2_SCORINING", True)

        # Output directory for Dual Output
        # output_dir = os.path.join(self.config.trainer.default_local_dir, f"jobid_{job_id}_outputs")
        output_dir = os.path.join(os.environ.get("PUZZLE_FEEDBACK_PATH", "./"), f"jobid_{job_id}")
        os.makedirs(output_dir, exist_ok=True)

        print(f"=== TRAINING CONFIG ===")
        print(f"Step-2 iterations per epoch: {step2_iterations}")
        print(f"Enable Step-2: {enable_step2}")
        print(f"Write Step Outputs: {write_step_outputs}")
        print(f"step2_mini_batch_iteration: {step2_mini_batch_iteration}")
        print(f"Output Directory: {output_dir}")
        print(f"Step-1 Scoring: {STEP1_SCORINING}")
        print(f"Step-2 Scoring: {STEP2_SCORINING}")
        print(f"======================")

        for epoch in range(self.config.trainer.total_epochs):
            os.environ["CURRENT_EPOCH"] = str(epoch+1)
            start = time.perf_counter()

            print(f"\n{'=' * 60}")
            print(f"Epoch {epoch + 1}: Starting Training Loop")
            print(f"{'=' * 60}")
            num_gen_batches = 0
            for batch_dict in self.train_dataloader:
                metrics = {}

                # ==================================
                # Step-1: Training
                # ==================================
                os.environ["STEP1_STATUS"] = "1"
                do_profile = (self.global_steps in self.config.trainer.profile_steps
                              if self.config.trainer.profile_steps is not None else False)
                with marked_timer("start_profile", timing_raw):
                    if do_profile:
                        self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
                        if self.use_reference_policy:
                            self.ref_policy_wg.start_profile()
                        if self.use_critic:
                            self.critic_wg.start_profile()
                        if self.use_rm:
                            self.rm_wg.start_profile()

                new_batch: DataProto = DataProto.from_single_dict(batch_dict)

                # pop those keys for generation
                if "multi_modal_data" in new_batch.non_tensor_batch.keys():
                    gen_batch = new_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                    )
                else:
                    gen_batch = new_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids"], )

                gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

                current_step = self.global_steps
                step1_reward_extra_infos = {}  # To store for Step 2
                step1_uids = None
                step1_non_tensor_batch = None  # Capture full non_tensor_batch for Step 2

                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, "red"):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with marked_timer("gen_max", timing_raw, "red"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                            new_batch = new_batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(new_batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)
                            new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))
                            new_batch.batch["reward_baselines"] = reward_baseline_tensor
                            del gen_baseline_batch, gen_baseline_output

                    new_batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object)
                    new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    new_batch = new_batch.union(gen_batch_output)
                    step1_uids = new_batch.non_tensor_batch["uid"]
                    step1_non_tensor_batch = deepcopy(new_batch.non_tensor_batch)  # Capture full non_tensor_batch

                    with marked_timer("reward", timing_raw, "yellow"):
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(new_batch)
                            new_batch = new_batch.union(reward_tensor)

                        # Step-1 Reward
                        reward_extra_infos_dict: dict[str, list]
                        try:
                            reward_result = self.reward_fn(new_batch, return_dict=True)
                            reward_tensor = reward_result["reward_tensor"]
                            reward_extra_infos_dict = reward_result.get("reward_extra_info", {})
                            step1_reward_extra_infos = reward_extra_infos_dict  # Capture for Step 2
                        except Exception as e:
                            print(f"Error in reward_fn: {e}")
                            reward_tensor = self.reward_fn(new_batch)
                            reward_extra_infos_dict = {}

                        new_batch.batch["token_level_scores"] = reward_tensor

                        if STEP1_SCORINING:
                            if reward_extra_infos_dict:
                                new_batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                            if self.config.algorithm.use_kl_in_reward:
                                new_batch, kl_metrics = apply_kl_penalty(
                                    new_batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                                )
                                metrics.update(kl_metrics)
                            else:
                                new_batch.batch["token_level_rewards"] = new_batch.batch["token_level_scores"]

                    if STEP1_SCORINING:
                        if not self.config.algorithm.filter_groups.enable:
                            batch = new_batch
                        else:
                            metric_name = self.config.algorithm.filter_groups.metric
                            if metric_name == "seq_final_reward":
                                new_batch.non_tensor_batch["seq_final_reward"] = (
                                    new_batch.batch["token_level_rewards"].sum(dim=-1).numpy()
                                )
                            elif metric_name == "seq_reward":
                                new_batch.non_tensor_batch["seq_reward"] = (
                                    new_batch.batch["token_level_scores"].sum(dim=-1).numpy()
                                )

                            prompt_uid2metric_vals = defaultdict(list)
                            for uid, metric_val in zip(
                                    new_batch.non_tensor_batch["uid"], new_batch.non_tensor_batch[metric_name], strict=True
                            ):
                                prompt_uid2metric_vals[uid].append(metric_val)

                            prompt_uid2metric_std = {}
                            for prompt_uid, metric_vals in prompt_uid2metric_vals.items():
                                prompt_uid2metric_std[prompt_uid] = np.std(metric_vals)

                            kept_prompt_uids = [
                                uid
                                for uid, std in prompt_uid2metric_std.items()
                                if std > 0 or len(prompt_uid2metric_vals[uid]) == 1
                            ]
                            num_prompt_in_batch += len(kept_prompt_uids)

                            kept_traj_idxs = []
                            for idx, traj_from_prompt_uid in enumerate(new_batch.non_tensor_batch["uid"]):
                                if traj_from_prompt_uid in kept_prompt_uids:
                                    kept_traj_idxs.append(idx)

                            new_batch = new_batch[kept_traj_idxs]
                            batch = new_batch if batch is None else DataProto.concat([batch, new_batch])

                            prompt_bsz = self.config.data.train_batch_size
                            if num_prompt_in_batch < prompt_bsz:
                                print(f"{num_prompt_in_batch=} < {prompt_bsz=}")
                                max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches
                                if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                                    print(f"{num_gen_batches=}. Keep generating...")
                                    progress_bar.update(1)
                                    continue
                                else:
                                    raise ValueError(
                                        f"{num_gen_batches=} >= {max_num_gen_batches=}."
                                        + " Generated too many. Please check if your data are too difficult."
                                    )
                            else:
                                traj_bsz = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
                                batch = batch[:traj_bsz]

                        # === Updating ===
                        batch.batch["response_mask"] = compute_response_mask(batch)

                        if self.config.trainer.balance_batch:
                            self._balance_batch(batch, metrics=metrics)

                        batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                        # recompute old_log_probs
                        with marked_timer("old_log_prob", timing_raw, "blue"):
                            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                            entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                            metrics.update({"actor/entropy": entropy_agg.detach().item()})
                            old_log_prob.batch.pop("entropys")
                            batch = batch.union(old_log_prob)

                        if self.use_reference_policy:
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                        if self.use_critic:
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                        with marked_timer("adv", timing_raw, "brown"):
                            norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                            batch = compute_advantage(batch, self.config.algorithm.adv_estimator,
                                                      self.config.algorithm.gamma, self.config.algorithm.lam,
                                                      self.config.actor_rollout_ref.rollout.n, norm_adv_by_std_in_grpo)

                        if self.use_critic:
                            critic_output = self.critic_wg.update_critic(batch)
                            metrics.update(reduce_metrics(critic_output.meta_info["metrics"]))

                        if self.config.trainer.critic_warmup <= current_step:
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                            metrics.update(reduce_metrics(actor_output.meta_info["metrics"]))

                        # Collect Metrics
                        metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                        metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                        if timing_raw["step"] > 0:
                            metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=self.resource_pool_manager.get_n_gpus()))
                        timing_raw = defaultdict(float)

                    # validate
                    if (self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and current_step % self.config.trainer.test_freq == 0):
                        with marked_timer("testing", timing_raw, "green"):
                            os.environ["VALID_STATUS"] = "1"
                            val_metrics: dict = self._validate()
                            os.environ["VALID_STATUS"] = "2"
                            val_metrics_tr: dict = self._validate_tr()
                            #if is_last_step:
                            #    last_val_metrics = val_metrics
                        metrics.update(val_metrics)
                        metrics.update(val_metrics_tr)
                        os.environ["VALID_STATUS"] = "0"

                    if self.config.trainer.save_freq > 0 and current_step % self.config.trainer.save_freq == 0:
                        self._save_checkpoint()
                    metrics["train/step"] = 1
                    metrics["train/step1/epoch"] = epoch + 1
                    print(json.dumps(metrics, indent=2, sort_keys=True))



                # ==================================
                # Dual Output (Parquet + JSONL)
                # ==================================
                if write_step_outputs:
                    # Collect data
                    step1_data = []
                    inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                    outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                    scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()

                    # Extract puzzle_ids if available
                    puzzle_ids = batch.non_tensor_batch.get("uid", ["unknown"] * len(inputs))  # Use UID or extract from extra_info

                    for i in range(len(inputs)):
                        record = {
                            "prompt": inputs[i],
                            "response": outputs[i],
                            "score": scores[i],
                            "epoch": epoch,
                            "global_step": self.global_steps,
                            "timestamp": time.time()
                        }
                        # Add extra info
                        if "step_verification_data" in step1_reward_extra_infos:
                            verif_data = step1_reward_extra_infos["step_verification_data"][i]
                            if isinstance(verif_data, dict):
                                record.update(verif_data)
                        step1_data.append(record)

                    # Write JSONL
                    jsonl_path = os.path.join(output_dir, f"step1_epoch_{epoch}_step_{self.global_steps}.jsonl")
                    with open(jsonl_path, "w", encoding="utf-8") as f:
                        for item in step1_data:
                            f.write(json.dumps(item, ensure_ascii=False) + "\n")

                    # Write Parquet
                    parquet_path = os.path.join(output_dir, f"step1_epoch_{epoch}_step_{self.global_steps}.parquet")
                    df_step1 = pd.DataFrame(step1_data)
                    # Convert complex columns to JSON strings to avoid PyArrow Mixed Type errors
                    for col in ["step1_json", "verifier_feedback", "clues", "extra_info", "step_verification_data"]:
                        if col in df_step1.columns:
                            df_step1[col] = df_step1[col].apply(lambda x: json.dumps(x, ensure_ascii=False) if x is not None else "")
                    df_step1.to_parquet(parquet_path)
                    print(f"Written Step-1 outputs to {jsonl_path} and {parquet_path}")

                # ==================================
                # Step-2: In-Memory Iteration
                # ==================================
                num_gen_batches = 0
                os.environ["STEP1_STATUS"] = "0"
                if enable_step2 and epoch >= 5 and step2_iterations > 0:
                    print(f"Starting Step-2 training for current batch...!")

                    file_path = os.path.join(os.environ.get("PUZZLE_FEEDBACK_PATH", "./"), f"jobid_{job_id}")
                    feedback_path = os.path.join(file_path, f"feedback_jobid_{job_id}_epoch_{epoch+1}.jsonl")

                    # feedback_data = []
                    # with open(feedback_path, "r", encoding="utf-8") as f:
                    #    for line in f:
                    #        line = line.strip()
                    #        if not line:
                    #            continue
                    #        feedback_data.append(json.loads(line))

                    feedback_data = datasets.load_dataset('json', data_files=feedback_path)['train']
                    process_train_fn = make_map_fn('train', 'our_zebra_puzzle_new_reward')
                    feedback_data = feedback_data.map(function=process_train_fn, with_indices=True)

                    train_feedback_path = save_dataset(
                        dataset=feedback_data,
                        output_dir=file_path,
                        filename=f"jobid_{job_id}_feedback",
                        sample_size=len(feedback_data)
                    )

                    self._create_dataloader_feedback(data_path=train_feedback_path, collate_fn=None, train_sampler=None)
                    # print("FEEDBACK DATALOADER DONE")
                    step2_iter = 0
                    for batch_dict in self.feedback_dataloader:
                        print(f"Batch size of step-2 batch = {batch_dict["input_ids"].shape[0]}")
                        step2_iter += 1
                        feedback_metrics = {}

                        do_profile = (self.global_steps in self.config.trainer.profile_steps
                                      if self.config.trainer.profile_steps is not None else False)
                        with marked_timer("start_profile", timing_raw):
                            if do_profile:
                                self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
                                if self.use_reference_policy:
                                    self.ref_policy_wg.start_profile()
                                if self.use_critic:
                                    self.critic_wg.start_profile()
                                if self.use_rm:
                                    self.rm_wg.start_profile()

                        new_batch: DataProto = DataProto.from_single_dict(batch_dict)
                        num_gen_batches += 1
                        # pop those keys for generation
                        if "multi_modal_data" in new_batch.non_tensor_batch.keys():
                            gen_batch = new_batch.pop(
                                batch_keys=["input_ids", "attention_mask", "position_ids"],
                                non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                            )
                        else:
                            gen_batch = new_batch.pop(
                                batch_keys=["input_ids", "attention_mask", "position_ids"],
                                non_tensor_batch_keys=["raw_prompt_ids"], )

                        # gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                        # is_last_step = self.global_steps >= self.total_training_steps

                        with marked_timer("step", timing_raw):
                            # generate a batch
                            with marked_timer("gen", timing_raw, "red"):
                                gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                                timing_raw.update(gen_batch_output.meta_info["timing"])
                                gen_batch_output.meta_info.pop("timing", None)

                            if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                                with marked_timer("gen_max", timing_raw, "red"):
                                    gen_baseline_batch = deepcopy(gen_batch)
                                    gen_baseline_batch.meta_info["do_sample"] = False
                                    gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                                    new_batch = new_batch.union(gen_baseline_output)
                                    reward_baseline_tensor = self.reward_fn(new_batch)
                                    reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                                    new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                                    new_batch.batch["reward_baselines"] = reward_baseline_tensor

                                    del gen_baseline_batch, gen_baseline_output

                            new_batch.non_tensor_batch["uid"] = np.array(
                                [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
                            )
                            # repeat to align with repeated responses in rollout
                            # new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                            new_batch = new_batch.union(gen_batch_output)

                            with marked_timer("reward", timing_raw, "yellow"):
                                # compute scores. Support both model and function-based.
                                # We first compute the scores using reward model. Then, we call reward_fn to combine
                                # the results from reward model and rule-based results.
                                if self.use_rm:
                                    # we first compute reward model score
                                    reward_tensor = self.rm_wg.compute_rm_score(new_batch)
                                    new_batch = new_batch.union(reward_tensor)

                                # we combine with rule-based rm
                                reward_extra_infos_dict: dict[str, list]
                                try:
                                    reward_result = self.reward_fn(new_batch, return_dict=True)
                                    reward_tensor = reward_result["reward_tensor"]
                                    reward_extra_infos_dict = reward_result.get("reward_extra_info", {})
                                except Exception as e:
                                    print(f"Error in reward_fn: {e}")
                                    reward_tensor = self.reward_fn(new_batch)
                                    reward_extra_infos_dict = {}

                                new_batch.batch["token_level_scores"] = reward_tensor


                                if STEP2_SCORINING:

                                    if reward_extra_infos_dict:
                                        new_batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                                    # compute rewards. apply_kl_penalty if available
                                    if self.config.algorithm.use_kl_in_reward:
                                        new_batch, kl_metrics = apply_kl_penalty(
                                            new_batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                                        )
                                        feedback_metrics.update(
                                            kl_metrics
                                        )  # TODO: This will be cleared if we use multiple genenration batches
                                    else:
                                        new_batch.batch["token_level_rewards"] = new_batch.batch["token_level_scores"]

                            if STEP2_SCORINING:
                                if not self.config.algorithm.filter_groups.enable:
                                    batch = new_batch
                                else:  # NOTE: When prompts after filtering is less than train batch size,
                                    # we skip to the next generation batch
                                    metric_name = self.config.algorithm.filter_groups.metric
                                    if metric_name == "seq_final_reward":
                                        # Turn to numpy for easier filtering
                                        new_batch.non_tensor_batch["seq_final_reward"] = (
                                            new_batch.batch["token_level_rewards"].sum(dim=-1).numpy()
                                        )
                                    elif metric_name == "seq_reward":
                                        new_batch.non_tensor_batch["seq_reward"] = (
                                            new_batch.batch["token_level_scores"].sum(dim=-1).numpy()
                                        )

                                    # Collect the sequence reward for each trajectory
                                    prompt_uid2metric_vals = defaultdict(list)
                                    for uid, metric_val in zip(
                                            new_batch.non_tensor_batch["uid"], new_batch.non_tensor_batch[metric_name], strict=True
                                    ):
                                        prompt_uid2metric_vals[uid].append(metric_val)

                                    prompt_uid2metric_std = {}
                                    for prompt_uid, metric_vals in prompt_uid2metric_vals.items():
                                        prompt_uid2metric_std[prompt_uid] = np.std(metric_vals)

                                    kept_prompt_uids = [
                                        uid
                                        for uid, std in prompt_uid2metric_std.items()
                                        if std > 0 or len(prompt_uid2metric_vals[uid]) == 1
                                    ]
                                    num_prompt_in_batch += len(kept_prompt_uids)

                                    kept_traj_idxs = []
                                    for idx, traj_from_prompt_uid in enumerate(new_batch.non_tensor_batch["uid"]):
                                        if traj_from_prompt_uid in kept_prompt_uids:
                                            kept_traj_idxs.append(idx)

                                    new_batch = new_batch[kept_traj_idxs]
                                    batch = new_batch if batch is None else DataProto.concat([batch, new_batch])

                                    prompt_bsz = self.config.data.train_batch_size
                                    if num_prompt_in_batch < prompt_bsz:
                                        print(f"{num_prompt_in_batch=} < {prompt_bsz=}")
                                        max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches
                                        if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                                            print(f"{num_gen_batches=}. Keep generating...")
                                            progress_bar.update(1)
                                            continue
                                        else:
                                            raise ValueError(
                                                f"{num_gen_batches=} >= {max_num_gen_batches=}."
                                                + " Generated too many. Please check if your data are too difficult."
                                                + " You could also try set max_num_gen_batches=0 to enable endless trials."
                                            )
                                    else:
                                        # Align the batch
                                        traj_bsz = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
                                        batch = batch[:traj_bsz]

                                # === Updating ===

                                batch.batch["response_mask"] = compute_response_mask(batch)

                                # Balance the number of valid tokens across DP ranks.
                                # NOTE: This usually changes the order of data in the `batch`,
                                # which won't affect the advantage calculation (since it's based on uid),
                                # but might affect the loss calculation (due to the change of mini-batching).
                                # TODO: Decouple the DP balancing and mini-batching.
                                if self.config.trainer.balance_batch:
                                    self._balance_batch(batch, metrics=feedback_metrics)

                                # compute global_valid tokens
                                batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                                # recompute old_log_probs
                                with marked_timer("old_log_prob", timing_raw, "blue"):
                                    old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                                    entropys = old_log_prob.batch["entropys"]
                                    response_masks = batch.batch["response_mask"]
                                    loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                                    entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                                    old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                                    feedback_metrics.update(old_log_prob_metrics)
                                    old_log_prob.batch.pop("entropys")
                                    batch = batch.union(old_log_prob)

                                if self.use_reference_policy:
                                    # compute reference log_prob
                                    with marked_timer("ref", timing_raw, "olive"):
                                        ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                                        batch = batch.union(ref_log_prob)

                                # compute values
                                if self.use_critic:
                                    with marked_timer("values", timing_raw, "cyan"):
                                        values = self.critic_wg.compute_values(batch)
                                        batch = batch.union(values)

                                with marked_timer("adv", timing_raw, "brown"):
                                    # compute advantages, executed on the driver process
                                    norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                                    batch = compute_advantage(
                                        batch,
                                        adv_estimator=self.config.algorithm.adv_estimator,
                                        gamma=self.config.algorithm.gamma,
                                        lam=self.config.algorithm.lam,
                                        num_repeat=self.config.actor_rollout_ref.rollout.n,
                                        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                                    )

                                # update critic
                                if self.use_critic:
                                    with marked_timer("update_critic", timing_raw, "pink"):
                                        critic_output = self.critic_wg.update_critic(batch)
                                    critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                                    feedback_metrics.update(critic_output_metrics)

                                # implement critic warmup
                                if self.config.trainer.critic_warmup <= self.global_steps:
                                    # update actor
                                    with marked_timer("update_actor", timing_raw, "red"):
                                        actor_output = self.actor_rollout_wg.update_actor(batch)
                                    actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                                    feedback_metrics.update(actor_output_metrics)

                            # collect metrics
                            feedback_metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                            feedback_metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                            # TODO: implement actual tflpo and theoretical tflpo
                            n_gpus = self.resource_pool_manager.get_n_gpus()
                            if timing_raw["step"] > 0:
                                feedback_metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                            timing_raw = defaultdict(float)  # clear timing

                            feedback_metrics["train/num_gen_batches"] = num_gen_batches
                            feedback_metrics["train/step"] = 2
                            feedback_metrics["train/step2/mini-batch_index"] = step2_iter
                            feedback_metrics["train/epoch"] = epoch + 1
                            print(json.dumps(feedback_metrics, indent=2, sort_keys=True))

                            batch = None
                            num_prompt_in_batch = 0
                            num_gen_batches = 0

                            if step2_iter >= step2_mini_batch_iteration:
                                break
                    # Validate & Save
                    if self.val_reward_fn and self.config.trainer.test_freq > 0 and current_step % self.config.trainer.test_freq == 0:
                        os.environ["VALID_STATUS"] = "3"
                        feedback_metrics.update(self._validate())
                        os.environ["VALID_STATUS"] = "4"
                        feedback_metrics.update(self._validate_tr())
                        print(json.dumps(feedback_metrics, indent=2, sort_keys=True))
                        os.environ["VALID_STATUS"] = "0"

                    try:
                        #os.remove(feedback_path)
                        os.remove(train_feedback_path)
                        print(f"Successfully removed {feedback_path}")
                        print(f"Successfully removed {train_feedback_path}")
                    except Exception as e:
                        print('Failed to remove feedback path: {}'.format(feedback_path))
                        print('Failed to remove feedback path: {}'.format(train_feedback_path))
                progress_bar.update(1)
                self.global_steps += 1

            # End of Epoch
            print(f"Epoch {epoch + 1} Completed.")

        print("Training Completed.")
        progress_bar.close()
