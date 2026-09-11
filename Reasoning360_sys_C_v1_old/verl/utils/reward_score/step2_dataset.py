import os
import json
from typing import Dict, Any, List, Optional
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import torch

from .step2_data_processor import load_tokenizer, step2_prompt_to_ids


class Step2Dataset(Dataset):
    """
    步骤2数据的PyTorch Dataset
    """
    
    def __init__(
        self,
        jsonl_path: str,
        tokenizer: Optional[AutoTokenizer] = None,
        model_path: Optional[str] = None,
        max_prompt_length: int = 4096,
        apply_chat_template: bool = True,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None
    ):
        """
        初始化Step2Dataset
        
        Args:
            jsonl_path: JSONL文件路径
            tokenizer: 加载好的tokenizer，如果为None则使用model_path加载
            model_path: 模型路径或名称，用于加载tokenizer
            max_prompt_length: 最大提示长度
            apply_chat_template: 是否应用chat template
            tokenizer_kwargs: 传递给load_tokenizer的额外参数
        """
        self.jsonl_path = jsonl_path
        self.max_prompt_length = max_prompt_length
        self.apply_chat_template = apply_chat_template
        
        # 加载tokenizer
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            if model_path is None:
                raise ValueError("Either tokenizer or model_path must be provided")
            tokenizer_kwargs = tokenizer_kwargs or {}
            self.tokenizer = load_tokenizer(model_path, **tokenizer_kwargs)
        
        # 加载数据
        self.data = self._load_data()
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """
        加载JSONL数据
        
        Returns:
            数据列表
        """
        data = []
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return data
    
    def __len__(self) -> int:
        """
        返回数据集中样本的数量
        
        Returns:
            样本数量
        """
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        获取指定索引的样本
        
        Args:
            idx: 样本索引
        
        Returns:
            样本数据，包含原始数据和tokenized数据
        """
        sample = self.data[idx]
        prompt = sample["prompt"]
        
        # 将提示转换为token ids
        token_ids, prompt_len = step2_prompt_to_ids(
            self.tokenizer,
            prompt,
            self.max_prompt_length,
            self.apply_chat_template
        )
        
        # 准备返回数据
        return {
            "raw_data": sample,
            "token_ids": token_ids,
            "prompt_len": prompt_len,
            "prompt": prompt,
            "id": sample.get("id", idx)
        }
    
    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        批次处理函数，用于DataLoader
        
        Args:
            batch: 样本列表
        
        Returns:
            批次数据，包含input_ids、attention_mask和prompt_lens
        """
        # 计算批次中的最大长度
        max_len = max(len(item["token_ids"]) for item in batch)
        
        input_ids = []
        attention_masks = []
        prompt_lens = []
        ids = []
        
        for item in batch:
            token_ids = item["token_ids"]
            prompt_len = item["prompt_len"]
            
            # 计算需要填充的长度
            pad_len = max_len - len(token_ids)
            
            # 填充输入ids
            padded_ids = token_ids + [self.tokenizer.pad_token_id] * pad_len
            input_ids.append(padded_ids)
            
            # 创建注意力掩码
            attention_mask = [1] * len(token_ids) + [0] * pad_len
            attention_masks.append(attention_mask)
            
            # 保存prompt长度和id
            prompt_lens.append(prompt_len)
            ids.append(item["id"])
        
        # 转换为tensor
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_masks, dtype=torch.bool)
        prompt_lens = torch.tensor(prompt_lens, dtype=torch.long)
        # ids保留为原始字符串类型，不转换为tensor
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "prompt_lens": prompt_lens,
            "ids": ids
        }


def create_step2_dataloader(
    jsonl_path: str,
    tokenizer: Optional[AutoTokenizer] = None,
    model_path: Optional[str] = None,
    batch_size: int = 16,
    max_prompt_length: int = 4096,
    apply_chat_template: bool = True,
    shuffle: bool = True,
    num_workers: int = 4,
    tokenizer_kwargs: Optional[Dict[str, Any]] = None,
    **dataloader_kwargs
) -> DataLoader:
    """
    创建步骤2数据的DataLoader
    
    Args:
        jsonl_path: JSONL文件路径
        tokenizer: 加载好的tokenizer，如果为None则使用model_path加载
        model_path: 模型路径或名称，用于加载tokenizer
        batch_size: 批次大小
        max_prompt_length: 最大提示长度
        apply_chat_template: 是否应用chat template
        shuffle: 是否打乱数据
        num_workers: 数据加载的进程数
        tokenizer_kwargs: 传递给load_tokenizer的额外参数
        **dataloader_kwargs: 传递给DataLoader的额外参数
    
    Returns:
        配置好的DataLoader
    """
    # 创建Dataset
    dataset = Step2Dataset(
        jsonl_path=jsonl_path,
        tokenizer=tokenizer,
        model_path=model_path,
        max_prompt_length=max_prompt_length,
        apply_chat_template=apply_chat_template,
        tokenizer_kwargs=tokenizer_kwargs
    )
    
    # 创建DataLoader
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=dataset.collate_fn,
        **dataloader_kwargs
    )
    
    return dataloader


def main():
    """
    测试Step2Dataset和create_step2_dataloader功能
    """
    # 测试数据路径
    test_data_path = "/path/to/step2_data.jsonl"
    model_path = "Qwen/Qwen2.5-1.5B-Instruct"
    batch_size = 2
    
    # 测试Dataset
    print(f"Testing Step2Dataset with {test_data_path}...")
    if os.path.exists(test_data_path):
        dataset = Step2Dataset(
            jsonl_path=test_data_path,
            model_path=model_path
        )
        
        print(f"Dataset length: {len(dataset)}")
        
        # 测试__getitem__
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"\nSample 0:")
            print(f"  ID: {sample['id']}")
            print(f"  Prompt length: {sample['prompt_len']}")
            print(f"  Token IDs length: {len(sample['token_ids'])}")
            print(f"  First 10 token IDs: {sample['token_ids'][:10]}")
        
        # 测试DataLoader
        print(f"\nTesting DataLoader with batch_size={batch_size}...")
        dataloader = create_step2_dataloader(
            jsonl_path=test_data_path,
            model_path=model_path,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0  # 单进程，方便调试
        )
        
        for batch_idx, batch in enumerate(dataloader):
            print(f"\nBatch {batch_idx}:")
            print(f"  input_ids shape: {batch['input_ids'].shape}")
            print(f"  attention_mask shape: {batch['attention_mask'].shape}")
            print(f"  prompt_lens: {batch['prompt_lens']}")
            print(f"  ids: {batch['ids']}")
            
            # 只测试一个批次
            if batch_idx == 0:
                break
    else:
        print(f"Test data path {test_data_path} does not exist")
    
    print("\nDataset and DataLoader test completed!")


if __name__ == "__main__":
    main()
