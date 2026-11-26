import os
import sys
import json
import datasets
import argparse
from sklearn.model_selection import train_test_split

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from verl.utils.data_process.utils import set_seed, sample_dataset, save_dataset



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='/home/asif/data3/HF_cache/ZebraLogic/', help='Path to json file')
    parser.add_argument('--data_setting', default=None, help='Path to json file')
    parser.add_argument('--output_dir', default='/home/asif/data3/HF_cache/ZebraLogic/', help='Directory to save processed data')
    parser.add_argument('--shot', default='zero', help='Directory to save processed data')
    parser.add_argument('--hdfs_dir', default=None, help='HDFS directory (optional)')
    parser.add_argument('--train_size', type=float, default=0.6, help='Proportion of data for train set')
    parser.add_argument('--test_size', type=float, default=0.4, help='Proportion of data for test set')
    parser.add_argument('--data_source_train', default='our_zebra_puzzle_new_reward', help='Name of data source')
    parser.add_argument('--data_source_test', default='our_zebra_puzzle_new_reward_test', help='Name of data source')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--train_sample_size', type=int, default=None, help='Number of samples to use from train. If None, use all.')
    args = parser.parse_args()


    args.data_file = os.path.join(args.data_path, 'Zebra_Puzzle_small_320.json')

    with open(args.data_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    pid_to_puzzle = []
    for line in data:
        pid_to_puzzle.append({"id":line['id'], "puzzle":line['puzzle']})
        #pid_to_puzzle[line['id']] = line['puzzle']
        #pid_to_puzzle[line['id']+'_sol'] = line['solution']

    args.output_dir = os.path.join(args.output_dir, 'pid_to_puzzle_dic_wo_sol.json')

    with open(args.output_dir, "w", encoding="utf-8") as f:
        json.dump(pid_to_puzzle, f, indent=2, ensure_ascii=False)

