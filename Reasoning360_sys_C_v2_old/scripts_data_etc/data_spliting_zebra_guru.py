#!/usr/bin/env python
# coding: utf-8

# 

# In[33]:

### Set all paths...!
import os
import sys

import numpy as np
import pandas as pd
import polars as pl

from os import path

import pyarrow.parquet as pq
from datasets import Dataset



#offline_eval_dir = 'offline_eval'
#online_eval_dir = 'online_eval'
train_dir = 'train'
test_dir = 'test'


# In[34]:

def add_suffix(filename, sample_size):
    if sample_size < 1000:
        size_str = f"{sample_size}"
    elif (sample_size / 1000) % 1 != 0:
        size_str = f"{sample_size / 1000:.1f}k"
    else:
        size_str = f"{sample_size // 1000}k"
    return f"{filename}_{size_str}"


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

    print('Processing small_training mediumn test')
    os.environ["HF_HOME"] = "/export/home/asifali/HF_cache"
    os.environ["HF_DATASETS_CACHE"] = "/export/home/asifali/HF_cache"

    base_dir = '/export/home/asifali/HF_cache/ZebraPuzzle_to_guru/small_train_med_test'
    new_dir = '/export/home/asifali/HF_cache/ZebraPuzzle_to_guru_10/small_train_med_test'

    ## Processing Train data sets...!
    directory = os.path.join(base_dir, train_dir)
    new_directory = os.path.join(new_dir, train_dir)

    print("directory = {}".format(directory))
    print()

    all_files = []
    rows_to_process = 10  # Total rows you want to keep

    for filename in sorted(os.listdir(directory)):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            print("Processing = {}".format(filename))
            if "livecodebench" in filepath:
                import polars as pl
                dataset = pl.read_parquet(filepath)
                dataset = dataset.to_pandas()
            else:
                dataset = pd.read_parquet(filepath)

            dataset_v1 = dataset.head(rows_to_process)
            df_reduced = Dataset.from_pandas(dataset_v1)
            test_output_path = save_dataset(dataset=df_reduced, output_dir=new_directory, filename_prefix=filename, sample_size=None)
            print(f"Data saved to {test_output_path}")


    ## Processing test data sets...!
    directory = os.path.join(base_dir, test_dir)
    new_directory = os.path.join(new_dir, test_dir)

    print("directory = {}".format(directory))
    print()

    all_files = []
    for filename in sorted(os.listdir(directory)):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            print("Processing = {}".format(filename))
            if "livecodebench" in filepath:
                import polars as pl
                dataset = pl.read_parquet(filepath)
                dataset = dataset.to_pandas()
            else:
                dataset = pd.read_parquet(filepath)

            dataset_v1 = dataset.head(rows_to_process)
            df_reduced = Dataset.from_pandas(dataset_v1)
            test_output_path = save_dataset(dataset=df_reduced, output_dir=new_directory, filename_prefix=filename, sample_size=None)
            print(f"Data saved to {test_output_path}")



    print('Processing Guru Adjusted data')
    base_dir = '/export/home/asifali/HF_cache/guru_data_adjusted'
    new_dir = '/export/home/asifali/HF_cache/guru_data_adjusted_10'

    ## Processing Train data sets...!
    directory = os.path.join(base_dir, train_dir)
    new_directory = os.path.join(new_dir, train_dir)

    print("directory = {}".format(directory))
    print()

    all_files = []
    for filename in sorted(os.listdir(directory)):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            print("Processing = {}".format(filename))
            if "livecodebench" in filepath:
                import polars as pl

                dataset = pl.read_parquet(filepath)
                dataset = dataset.to_pandas()
            else:
                dataset = pd.read_parquet(filepath)

            dataset_v1 = dataset.head(rows_to_process)
            df_reduced = Dataset.from_pandas(dataset_v1)
            test_output_path = save_dataset(dataset=df_reduced, output_dir=new_directory, filename_prefix=filename, sample_size=None)
            print(f"Data saved to {test_output_path}")

    ## Processing test data sets...!
    directory = os.path.join(base_dir, test_dir)
    new_directory = os.path.join(new_dir, test_dir)

    print("directory = {}".format(directory))
    print()

    all_files = []
    for filename in sorted(os.listdir(directory)):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            print("Processing = {}".format(filename))
            if "livecodebench" in filepath:
                import polars as pl

                dataset = pl.read_parquet(filepath)
                dataset = dataset.to_pandas()
            else:
                dataset = pd.read_parquet(filepath)

            dataset_v1 = dataset.head(rows_to_process)
            df_reduced = Dataset.from_pandas(dataset_v1)
            test_output_path = save_dataset(dataset=df_reduced, output_dir=new_directory, filename_prefix=filename, sample_size=None)
            print(f"Data saved to {test_output_path}")
