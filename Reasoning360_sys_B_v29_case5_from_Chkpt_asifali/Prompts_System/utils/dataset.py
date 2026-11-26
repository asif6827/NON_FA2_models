import json
import os
import pandas as pd
from typing import Dict, List, Any

def load_local_dataset(file_path: str) -> List[Dict[str, Any]]:
    """
    Load dataset from local file (JSON, JSONL, or Parquet format).
    
    Args:
        file_path: Path to the dataset file
        
    Returns:
        List of data items
    """
    data: List[Dict[str, Any]] = []
    print(f"[Data] Loading data from {file_path}...")
    try:
        if file_path.endswith('.jsonl'):
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
        elif file_path.endswith('.json'):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = json.load(f)
                if isinstance(content, list):
                    data = content
        elif file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
            data = df.to_dict('records')
        else:
            print(f"[Warning] Unsupported file format: {os.path.splitext(file_path)[1]}")
            return []
        print(f"[Data] Loaded {len(data)} samples.")
        return data
    except Exception as e:
        print(f"[Error] Failed to load data: {e}")
        return []

def get_puzzle_text(item: Dict[str, Any]) -> str:
    """
    Extract puzzle text from a data item.
    
    Args:
        item: Data item containing puzzle information
        
    Returns:
        Puzzle text as string
    """
    puzzle = item.get("puzzle")
    if isinstance(puzzle, str):
        return puzzle
    return ""
