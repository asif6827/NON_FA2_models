import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def main():
    parser = argparse.ArgumentParser(description="Merge Base Model and LoRA Adapter for vLLM Inference")
    parser.add_argument("--base_model_path", type=str, default="/root/autodl-tmp/model/Qwen2.5-0.5B-Instruct", help="Base 模型路径 (e.g. Qwen2.5-Math-1.5B)")
    parser.add_argument("--checkpoint_path", type=str, default="/root/autodl-tmp/math/SFT/sft-qwen2.5-0.5b-Instruct-checkpoints/checkpoint-6657", help="微调后的 Checkpoint 路径")
    parser.add_argument("--output_path", type=str, default="/root/autodl-tmp/math/SFT/sft-qwen2.5-0.5b-Instruct-checkpoints/full-6657", help="合并后模型的输出路径")
    parser.add_argument("--safe_serialization", action="store_true", default=True, help="是否使用 safetensors 保存")
    
    args = parser.parse_args()
    
    print(f"=== Starting Model Merge Process ===")
    print(f"Base Model: {args.base_model_path}")
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Output Dir: {args.output_path}")

    # 1. Detect Checkpoint Type (LoRA vs Full)
    is_lora = os.path.exists(os.path.join(args.checkpoint_path, "adapter_config.json"))
    
    if is_lora:
        print("[Info] Detected LoRA adapter. Performing merge...")
        print("Loading Base Model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model_path,
            torch_dtype=torch.float16,
            device_map="auto", 
            trust_remote_code=True
        )
        
        print("Loading LoRA Adapter...")
        model = PeftModel.from_pretrained(base_model, args.checkpoint_path)
        
        print("Merging weights (merge_and_unload)...")
        model = model.merge_and_unload()
    else:
        print("[Info] No LoRA config found. Assuming Full Fine-Tuning checkpoint.")
        print("Loading Checkpoint directly...")
        model = AutoModelForCausalLM.from_pretrained(
            args.checkpoint_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

    print(f"Saving merged model to {args.output_path}...")
    model.save_pretrained(args.output_path, safe_serialization=args.safe_serialization)

    print("Handling Tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path, trust_remote_code=True)
    except:
        print("Tokenizer not found in checkpoint, loading from base...")
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True)
    
    tokenizer.save_pretrained(args.output_path)
    
    print(f"=== Success! Model saved to {args.output_path} ===")
    print(f"Now you can run generate_rollouts.py with --model_path {args.output_path}")

if __name__ == "__main__":
    main()


