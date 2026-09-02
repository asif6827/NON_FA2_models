#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SFT training script for logic puzzle solving.
Supports multiple models, full fine-tuning, and LoRA.
"""
import os
import sys
import logging
import torch
import json
import shutil
import argparse
from tqdm import tqdm
from datetime import datetime
from dataclasses import dataclass, field

import transformers
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
)
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, TaskType

job_id = os.getenv("SLURM_JOB_ID")
print("SLURM Job ID:", job_id)


logger = logging.getLogger(__name__)

def setup_logging(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_file = os.path.join(output_dir, "train.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    )
    logger.addHandler(file_handler)

    logger.setLevel(logging.INFO)
    transformers.utils.logging.set_verbosity_info()
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"Logs will be written to: {log_file}")

def keep_top_k_checkpoints(output_dir: str, k: int = 2,
                           metric_name: str = "eval_loss",
                           greater_is_better: bool = False):
    state_path = os.path.join(output_dir, "trainer_state.json")
    if not os.path.exists(state_path):
        logger.warning(f"Could not find {state_path}, cannot perform top-{k} checkpoint cleanup.")
        return

    with open(state_path, "r") as f:
        state = json.load(f)

    log_history = state.get("log_history", [])
    records = []
    for entry in log_history:
        if metric_name in entry and "step" in entry:
            step = entry["step"]
            value = entry[metric_name]
            records.append((step, value))

    if len(records) == 0:
        logger.warning(f"No {metric_name} records found in log_history, skipping top-{k} cleanup.")
        return

    if len(records) <= k:
        logger.info(f"Number of eval runs is {len(records)} <= {k}, no need to clean up checkpoints.")
        return

    records_sorted = sorted(records, key=lambda x: x[1], reverse=greater_is_better)
    best_steps = [r[0] for r in records_sorted[:k]]
    best_ckpt_names = {f"checkpoint-{step}" for step in best_steps}

    logger.info(f"Top-{k} checkpoints selected based on {metric_name}: {best_ckpt_names}")

    for name in os.listdir(output_dir):
        full_path = os.path.join(output_dir, name)
        if os.path.isdir(full_path) and name.startswith("checkpoint-"):
            if name not in best_ckpt_names:
                logger.info(f"Deleting non-optimal checkpoint: {name}")
                shutil.rmtree(full_path, ignore_errors=True)

    logger.info(f"top-{k} checkpoint cleanup completed.")


def main():
    parser = argparse.ArgumentParser(description="SFT Training Script for Logic Puzzle Solving")
    
    parser.add_argument("--model_name_or_path", type=str, default="/export/home/asifali/HF_cache/Qwen2.5-7B-Instruct", help="Local model folder path or Hugging Face model ID")
    parser.add_argument("--select_", type=str, default="combined", help="Training data path (json/jsonl, containing messages field)")
    parser.add_argument("--data_path", type=str, default="/export/home/asifali/Reasoning360/Prompts_SFT_data", help="Path to training data folders")
    parser.add_argument("--save_dir", type=str, default="/export/home/asifali/Reasoning360/Prompts_Checkpoint/", help="Model save root directory")
    
    # LoRA Configuration
    parser.add_argument("--use_lora", action="store_true", help="Whether to use LoRA")
    parser.add_argument("--lora_r", type=int, default=64,  help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout rate")
    
    # Training Configuration
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs")
    #parser.add_argument("--eval_steps", type=int, default=1000, help="Steps for Evaluation")
    #parser.add_argument("--save_steps", type=int, default=1500, help="Steps for saving")
    
    # Logging Configuration
    parser.add_argument("--wandb_project", type=str, default="Logic-Puzzle-SFT", help="WandB project name")
    parser.add_argument("--wandb_run_name", type=str, default="sft_logic_puzzle", help="WandB experiment run name")
    
    args = parser.parse_args()
    data_id = args.data_path.split('_job')[-1]
    if args.select_ == "combined":
        args.eval_data_path = os.path.join(args.data_path, "jsonl/test.jsonl")
        args.data_path = os.path.join(args.data_path, "jsonl/train.jsonl")

    elif args.select_ == "correct_only":
        args.eval_data_path = os.path.join(args.data_path, "jsonl/correct_only/test.jsonl")
        args.data_path = os.path.join(args.data_path,"jsonl/correct_only/train.jsonl")

    elif args.select_ == "incorrect_only":
        args.eval_data_path = os.path.join(args.data_path, "jsonl/incorrect_only/test.jsonl")
        args.data_path = os.path.join(args.data_path,"jsonl/incorrect_only/train.jsonl")

    elif args.select_ == "mixed":
        args.eval_data_path = os.path.join(args.data_path, "jsonl/correct_only/test.jsonl")
        args.data_path = os.path.join(args.data_path,"jsonl/train.jsonl")


    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    
    args.save_dir = os.path.join(args.save_dir, f"data_job_{data_id}_epoch_{args.num_train_epochs}_data_{args.select_}_{timestamp}_jobid_{job_id}")
    print()
    print("Arguments = {}".format(args))
    print()
    sft_config = SFTConfig(
        output_dir=args.save_dir,
        logging_steps=1,
        save_strategy="epoch",
        save_total_limit=args.num_train_epochs,  # keep all Epochs
        save_only_model=True,
        eval_strategy="epoch",
        do_eval=True,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        num_train_epochs=args.num_train_epochs,
        learning_rate=2e-5,
        no_cuda=False,
        bf16=True,
        fp16=False,
        save_safetensors=True,
        gradient_checkpointing=True,
        report_to=["wandb"],
        run_name=args.wandb_run_name,
        weight_decay=0.1,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        max_length=4096,
        packing=False,
        max_steps=-1,
        dataloader_drop_last=False,
        dataset_text_field=None,
        assistant_only_loss=True,
        load_best_model_at_end=False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    os.environ["WANDB_PROJECT"] = args.wandb_project
    os.environ["WANDB_NAME"] = args.wandb_run_name

    setup_logging(sft_config.output_dir)
    logger.info(f"Training parameters: {args}")
    logger.info(f"SFTConfig: {sft_config}")

    set_seed(42)
    try:
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        pass

    logger.info(f"Versions - torch: {torch.__version__}, transformers: {transformers.__version__}")

    logger.info(f"Loading Tokenizer: {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Tokenizer eos_token: {tokenizer.eos_token}, pad_token: {tokenizer.pad_token}")

    tmpl = getattr(tokenizer, "chat_template", None)
    if tmpl and "{% generation %}" in tmpl:
        sft_config.assistant_only_loss = True
        logger.info("Detected chat_template contains generation block, assistant_only_loss=True")
    else:
        sft_config.assistant_only_loss = False
        logger.info("chat_template does not contain generation block, disabled assistant_only_loss (full sequence loss)")

    logger.info("Loading model...")
    _use_cuda = torch.cuda.is_available()
    if _use_cuda and torch.cuda.is_bf16_supported():
        _dtype = torch.bfloat16
    elif _use_cuda:
        _dtype = torch.float16
    else:
        _dtype = torch.float32

    _attn = "eager"
    if _use_cuda:
        try:
            import flash_attn  # noqa: F401
            _attn = "flash_attention_2"
        except Exception:
            _attn = "sdpa"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=_dtype,
        attn_implementation=_attn,
        trust_remote_code=True,
        use_cache=False if sft_config.gradient_checkpointing else True,
    )

    first_attn = model.model.layers[0].self_attn.__class__.__name__
    logger.info(f"Self attention impl: {first_attn}")
    
    peft_config = None
    if args.use_lora:
        logger.info("Detected LoRA mode enabled")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )
        sft_config.learning_rate = 3e-4
        logger.info("Adjusted learning rate to 3e-4 in LoRA mode")


    logger.info(f"Loading dataset: {args.data_path}")
    dataset = load_dataset("json", data_files=args.data_path, split="train")
    logger.info(f"Training dataset loaded, total {len(dataset)} samples")
    

    eval_dataset = load_dataset("json", data_files=args.eval_data_path, split="train")
    logger.info(f"Validation dataset loaded, total {len(eval_dataset)} samples")
    train_dataset = dataset

    logger.info(f"Training set: {len(train_dataset)} samples, Validation set: {len(eval_dataset)} samples")

    logger.info("*** Data Sanity Check (First 2 Examples) ***")
    for i in range(min(2, len(train_dataset))):
        sample = train_dataset[i]
        if "messages" not in sample:
            raise ValueError(f"`messages` field not found in sample {i}")
        formatted = tokenizer.apply_chat_template(
            sample["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        logger.info(f"Sample {i}:\n{formatted[:500]}...\n{'=' * 40}")

    logger.info("Building SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    train_dataloader = trainer.get_train_dataloader()
    logger.info(f"Train dataloader steps per epoch: {len(train_dataloader)}, num_train_epochs: {sft_config.num_train_epochs}")

    # Start Training
    logger.info("Starting training process...")
    sft_config.do_train = True
    train_result = trainer.train()

    # Save and Report
    logger.info(f"Training completed, saving model to {sft_config.output_dir}")
    #trainer.save_model(sft_config.output_dir)
    #tokenizer.save_pretrained(sft_config.output_dir)

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    with open(os.path.join(sft_config.output_dir, "script_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)
    

    with open(os.path.join(sft_config.output_dir, "sft_config.json"), "w") as f:
        sft_config_dict = {}
        for k, v in sft_config.__dict__.items():
            if not k.startswith('_'):
                try:
                    json.dumps(v)
                    sft_config_dict[k] = v
                except (TypeError, ValueError):
                    if hasattr(v, '__dict__'):
                        sft_config_dict[k] = f"{v.__class__.__name__} object"
                    else:
                        sft_config_dict[k] = str(v)
        json.dump(sft_config_dict, f, indent=2, ensure_ascii=False)

    #keep_top_k_checkpoints(output_dir=sft_config.output_dir, k=2, metric_name="eval_loss", greater_is_better=False,)

    logger.info("All steps completed.")


if __name__ == "__main__":
    main()