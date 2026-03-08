import os
import json
from typing import Dict, Any, List, Tuple, Optional
from transformers import AutoTokenizer
import torch


def load_tokenizer(model_path: str, **kwargs) -> AutoTokenizer:
    """
    加载tokenizer
    
    Args:
        model_path: 模型路径或名称
        **kwargs: 传递给AutoTokenizer.from_pretrained的额外参数
    
    Returns:
        加载好的tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path, **kwargs)
    
    # 设置默认参数
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if tokenizer.chat_template is None:
        # 设置默认的chat template
        tokenizer.chat_template = "{system_message}\n\n{user_message}\n\n{assistant_message}"
    
    return tokenizer


def step2_prompt_to_ids(
    tokenizer: AutoTokenizer,
    prompt: List[Dict[str, str]],
    max_prompt_length: int = 4096,
    apply_chat_template: bool = True
) -> Tuple[List[int], int]:
    """
    将步骤2提示转换为token ids
    
    Args:
        tokenizer: 加载好的tokenizer
        prompt: 提示内容，格式为[{"role": "user", "content": "..."}]
        max_prompt_length: 最大提示长度
        apply_chat_template: 是否应用chat template
    
    Returns:
        tuple: (token_ids, prompt_length)
    """
    if apply_chat_template:
        # 使用chat template格式化提示
        formatted_prompt = tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=True  # 添加生成提示
        )
        
        # 将格式化后的文本转换为token ids
        token_ids = tokenizer.encode(
            formatted_prompt,
            max_length=max_prompt_length,
            truncation=True,
            add_special_tokens=True
        )
    else:
        # 直接使用第一个user的content
        content = prompt[0]["content"] if prompt and prompt[0]["role"] == "user" else ""
        
        # 转换为token ids
        token_ids = tokenizer.encode(
            content,
            max_length=max_prompt_length,
            truncation=True,
            add_special_tokens=True
        )
    
    return token_ids, len(token_ids)


def batch_step2_prompt_to_ids(
    tokenizer: AutoTokenizer,
    prompts: List[List[Dict[str, str]]],
    max_prompt_length: int = 4096,
    apply_chat_template: bool = True,
    padding: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    批量将步骤2提示转换为token ids
    
    Args:
        tokenizer: 加载好的tokenizer
        prompts: 提示列表，每个提示格式为[{"role": "user", "content": "..."}]
        max_prompt_length: 最大提示长度
        apply_chat_template: 是否应用chat template
        padding: 是否进行padding
    
    Returns:
        tuple: (input_ids, attention_mask, prompt_lens)
    """
    all_token_ids = []
    all_prompt_lens = []
    
    # 转换每个提示
    for prompt in prompts:
        token_ids, prompt_len = step2_prompt_to_ids(
            tokenizer,
            prompt,
            max_prompt_length,
            apply_chat_template
        )
        all_token_ids.append(token_ids)
        all_prompt_lens.append(prompt_len)
    
    # 计算最大长度
    max_len = max(len(ids) for ids in all_token_ids) if padding else max_prompt_length
    
    # 进行padding
    padded_input_ids = []
    attention_masks = []
    
    for ids in all_token_ids:
        # 计算需要填充的长度
        pad_len = max_len - len(ids)
        
        # 填充输入ids
        padded_ids = ids + [tokenizer.pad_token_id] * pad_len
        padded_input_ids.append(padded_ids)
        
        # 创建注意力掩码
        attention_mask = [1] * len(ids) + [0] * pad_len
        attention_masks.append(attention_mask)
    
    # 转换为tensor
    input_ids = torch.tensor(padded_input_ids, dtype=torch.long)
    attention_mask = torch.tensor(attention_masks, dtype=torch.bool)
    prompt_lens = torch.tensor(all_prompt_lens, dtype=torch.long)
    
    return input_ids, attention_mask, prompt_lens


def load_step2_data(jsonl_path: str) -> List[Dict[str, Any]]:
    """
    加载步骤2生成的JSONL数据
    
    Args:
        jsonl_path: JSONL文件路径
    
    Returns:
        数据列表
    """
    data = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return data


def prepare_step2_batch(
    data: List[Dict[str, Any]],
    tokenizer: AutoTokenizer,
    max_prompt_length: int = 4096,
    apply_chat_template: bool = True,
    padding: bool = True
) -> Dict[str, torch.Tensor]:
    """
    准备步骤2数据批次
    
    Args:
        data: 数据列表
        tokenizer: 加载好的tokenizer
        max_prompt_length: 最大提示长度
        apply_chat_template: 是否应用chat template
        padding: 是否进行padding
    
    Returns:
        批次数据，包含input_ids、attention_mask和prompt_lens
    """
    # 提取提示
    prompts = [item["prompt"] for item in data]
    
    # 转换为ids
    input_ids, attention_mask, prompt_lens = batch_step2_prompt_to_ids(
        tokenizer,
        prompts,
        max_prompt_length,
        apply_chat_template,
        padding
    )
    
    # 准备批次数据
    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "prompt_lens": prompt_lens
    }
    
    # 如果有额外的标签信息，可以在这里添加
    # 例如：batch["labels"] = ...
    
    return batch


def main():
    """
    测试数据处理功能
    """
    # 测试数据路径
    test_data_path = "/path/to/step2_data.jsonl"
    model_path = "Qwen/Qwen2.5-1.5B-Instruct"
    
    # 加载tokenizer
    print(f"Loading tokenizer from {model_path}...")
    tokenizer = load_tokenizer(model_path)
    
    # 加载测试数据
    print(f"Loading test data from {test_data_path}...")
    if os.path.exists(test_data_path):
        data = load_step2_data(test_data_path)
        print(f"Loaded {len(data)} samples")
        
        # 测试单条数据处理
        if data:
            sample = data[0]
            prompt = sample["prompt"]
            print(f"\nSample prompt: {prompt}")
            
            token_ids, prompt_len = step2_prompt_to_ids(tokenizer, prompt)
            print(f"Token ids length: {len(token_ids)}, Prompt length: {prompt_len}")
            print(f"First 10 token ids: {token_ids[:10]}")
            
            # 测试批量数据处理
            if len(data) >= 2:
                batch_data = data[:2]
                batch = prepare_step2_batch(batch_data, tokenizer)
                print(f"\nBatch input_ids shape: {batch['input_ids'].shape}")
                print(f"Batch attention_mask shape: {batch['attention_mask'].shape}")
                print(f"Batch prompt_lens: {batch['prompt_lens']}")
    else:
        print(f"Test data path {test_data_path} does not exist")
        
    print("\nData processor test completed!")


if __name__ == "__main__":
    main()
