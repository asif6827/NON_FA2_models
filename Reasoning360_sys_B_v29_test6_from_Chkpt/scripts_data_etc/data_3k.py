#!/usr/bin/env python
# coding: utf-8

# 

# In[33]:

### Set all paths...!
import os
import re
from decimal import Decimal
import sys
import random

import numpy as np
import pandas as pd
import polars as pl

from os import path

import pyarrow.parquet as pq
from datasets import Dataset



### Load dataset from Hugging-face

# import json
# from datasets import load_dataset
# train_data = load_dataset("LLM360/guru-RL-92k", split="train", streaming=True)

# print(f"Columns: {train_data.column_names}")
# print()
# print(f"First item: {next(iter(train_data))}")
# In[31]:





# In[34]:

def add_suffix(filename, sample_size):
    if sample_size < 1000:
        size_str = f"{sample_size}"
    elif (sample_size / 1000) % 1 != 0:
        size_str = f"{sample_size / 1000:.1f}k"
    else:
        size_str = f"{sample_size // 1000}k"
    return f"{filename}_{size_str}"

def to_int_any(x, *, suffixes=None):
    """
    Convert numbers like:
      190      -> 190
      '54K'    -> 54000
      '1.8k'   -> 1800
      '2M'     -> 2000000
      '3.5b'   -> 3500000000
    Returns int; raises ValueError if unparseable.
    """
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        return int(Decimal(str(x)))

    if not isinstance(x, str):
        raise ValueError(f"Unsupported type: {type(x)}")

    s = x.strip().replace(',', '').replace(' ', '')
    m = re.fullmatch(r'([+-]?(?:\d+(?:\.\d+)?|\.\d+))([a-zA-Z]?)', s)
    if not m:
        raise ValueError(f"Cannot parse number: {x!r}")

    num = Decimal(m.group(1))
    suf = m.group(2).lower()

    # default suffix multipliers
    multipliers = {'k': 10**3, 'm': 10**6, 'b': 10**9, 't': 10**12}
    if suffixes:
        multipliers.update({k.lower(): v for k, v in suffixes.items()})

    factor = multipliers.get(suf, 1)  # no suffix -> factor 1
    return int(num * factor)


def sample_dataset(dataset, sample_size):
    """
    Sample a dataset to a given size.
    """
    if sample_size is not None:
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        indices = indices[:min(sample_size, len(dataset))]
        dataset = dataset.select(indices)
    return dataset



def save_dataset(dataset, output_dir, filename_prefix, sample_size=None):
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
    filename_prefix = filename_prefix.split('.parquet')[0]
    filename = add_suffix(filename_prefix, sample_size)
    output_path = os.path.join(output_dir, f"{filename}.parquet")

    # Save dataset
    dataset.to_parquet(output_path)

    return output_path


# In[36]:

if __name__ == "__main__":
    hp = False
    panther = True

    if hp:
        os.environ["HF_HOME"] = "/home/asif/data3/HF_Cache"
        os.environ["HF_DATASETS_CACHE"] = "/home/asif/data3/HF_Cache"
        os.environ["TRANSFORMERS_CACHE"] = "/home/asif/data3/HF_Cache"

        base_dir = '/home/asif/data3/HF_Cache/guru_data/'
        new_dir = '/home/asif/data3/HF_Cache/guru_data_3K/'
        #intermediate_dir = '/home/asif/data3/HF_Cache/guru_intermediate/'
    elif panther:
        os.environ["HF_HOME"] = "/export/home/asifali/HF_cache"
        os.environ["HF_DATASETS_CACHE"] = "/export/home/asifali/HF_cache"
        os.environ["TRANSFORMERS_CACHE"] = "/export/home/asifali/HF_cache"

        base_dir = '/export/home/asifali/HF_cache/guru_data/'
        new_dir = '/export/home/asifali/HF_cache/guru_data_3K/'
        #intermediate_dir = '/export/home/asifali/HF_cache/guru_intermediate'


    offline_eval_dir = 'offline_eval'
    online_eval_dir = 'online_eval'
    train_dir = 'train'
    rows_to_process = 3000
    ### Processing maths data...!
    directory = os.path.join(base_dir, train_dir)
    new_directory = os.path.join(new_dir, train_dir)


    print("directory = {}".format(directory))
    print()
    all_files = os.listdir(directory)
    all_domains = {}

    for filename in all_files:
        domain = filename.split('__')[0]  # 'codegen__leetcode2k_1.3k.parquet'
        all_domains[domain] = {}

    for filename in all_files:
        domain = filename.split('__')[0]  # 'codegen__leetcode2k_1.3k.parquet'
        all_domains[domain][filename] = to_int_any(filename.split('.parquet')[0].split('_')[-1])

    for domain, file_stat in all_domains.items():
        for fname, count in file_stat.items():
            filepath = os.path.join(directory, fname)
            if os.path.isfile(filepath):
                print("Processing = {}".format(fname))

                if "codegen__" in fname:
                    multiplier = 0.166297118
                    sample_size = int(multiplier * count)
                    if "livecodebench" in fname:
                        import polars as pl
                        dataset = pl.read_parquet(filepath)
                        dataset = dataset.to_pandas()
                    else:
                        dataset = pd.read_parquet(filepath)
                    dataset_v1 = dataset.sample(n=sample_size)

                elif "logic__" in fname:
                    multiplier = 0.476114902
                    dataset = pd.read_parquet(filepath)
                    sample_size = int(multiplier * count)
                    dataset_v1 = dataset.sample(n = sample_size)

                elif "simulation__" in fname:
                    multiplier = 0.810810811
                    dataset = pd.read_parquet(filepath)
                    sample_size = int(multiplier * count)
                    dataset_v1 = dataset.sample(n = sample_size)

                elif "table__" in fname:
                    multiplier = 0.517241379
                    dataset = pd.read_parquet(filepath)
                    sample_size = int(multiplier * count)
                    dataset_v1 = dataset.sample(n = sample_size)

                elif "stem__" in fname:
                    multiplier = 0.833333333
                    dataset = pd.read_parquet(filepath)
                    sample_size = int(multiplier * count)
                    dataset_v1 = dataset.sample(n = sample_size)


                elif "math__" in fname:
                    multiplier = 0.055555556
                    dataset = pd.read_parquet(filepath)
                    sample_size = int(multiplier * count)
                    dataset_v1 = dataset.sample(n = sample_size)


                df_reduced = Dataset.from_pandas(dataset_v1)
                test_output_path = save_dataset(dataset=df_reduced, output_dir=new_directory, filename_prefix=fname, sample_size=None)
                print(f"Data saved to {test_output_path}")