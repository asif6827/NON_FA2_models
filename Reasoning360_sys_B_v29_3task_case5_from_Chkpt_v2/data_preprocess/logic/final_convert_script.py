#!/usr/bin/env python3
"""
Convert the Parquet format zebra puzzle dataset to JSONL format.
"""

import pandas as pd
import json
import numpy as np
from datetime import datetime
import os

# Input and output paths
PARQUET_PATH = '/home/wwq416/snap/wwq/puzzle-asif/final_code-1/dataset/logic__zebra_puzzle_1.3k.parquet'
OUTPUT_PATH = '/home/wwq416/snap/wwq/puzzle-asif/final_code-1/dataset/converted_zebra_puzzle.jsonl'

# Create output directory if it doesn't exist
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

def to_list_if_array(obj):
    """Convert numpy array to list if needed."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def main():
    """Main function to convert Parquet to JSONL."""
    print(f"Reading Parquet file: {PARQUET_PATH}")
    df = pd.read_parquet(PARQUET_PATH)
    print(f"Total rows: {len(df)}")
    
    print(f"Writing to JSONL file: {OUTPUT_PATH}")
    
    success_count = 0
    
    with open(OUTPUT_PATH, 'w') as f:
        for index, row in df.iterrows():
            # Print progress every 100 rows
            if (index + 1) % 100 == 0:
                print(f"Processed {index + 1} rows... Success: {success_count}")
            
            try:
                # Get data
                puzzle_id = row['puzzle_id']
                config = row['config']
                instruction = row['instruction']
                clues = to_list_if_array(row['clues'])
                ground_truth = row['ground_truth'].copy()
                
                # Process ground truth - convert all numpy arrays to lists
                ground_truth['header'] = to_list_if_array(ground_truth['header'])
                ground_truth['rows'] = to_list_if_array(ground_truth['rows'])
                
                # Convert rows in ground truth (each row might be a numpy array)
                if isinstance(ground_truth['rows'], list):
                    ground_truth['rows'] = [to_list_if_array(row_item) for row_item in ground_truth['rows']]
                
                # Create puzzle text
                puzzle_text = instruction.strip()
                puzzle_text += "\n\n## Clues:"
                for i, clue in enumerate(clues, 1):
                    puzzle_text += f"\n{i}. {clue.strip()}"
                
                # Create output in the exact format of easy_size_data.jsonl
                output = {
                    "id": puzzle_id,
                    "size": f"{config['rows']}*{config['cols']}",
                    "puzzle": puzzle_text,
                    "solution": ground_truth,
                    "created_at": datetime.now().isoformat() + "Z"
                }
                
                # Convert output to JSON and write to file
                json_str = json.dumps(output, ensure_ascii=False)
                f.write(json_str + '\n')
                success_count += 1
                
            except Exception as e:
                # Skip failed rows
                continue
    
    print(f"Conversion completed. Output file: {OUTPUT_PATH}")
    print(f"Successfully processed {success_count}/{len(df)} rows")
    
    # Show a sample of the converted data
    print("\nSample entry:")
    with open(OUTPUT_PATH, 'r') as f:
        first_line = f.readline()
        if first_line:
            sample = json.loads(first_line)
            print(json.dumps(sample, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()