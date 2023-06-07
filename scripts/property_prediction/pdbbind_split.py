import os
import pickle
import random
import argparse
import torch
import numpy as np


def coretest_split(index_path, test_path, val_ratio=0.1, val_num=None):
    with open(index_path, 'rb') as f:
        index = pickle.load(f)
    test_ids = [f for f in os.listdir(test_path) if len(f) == 4]
    all_ids = [os.path.basename(i[0])[:4] for i in index]
    print('Original test set size: ', len(test_ids))
    test_index = [all_ids.index(test_id) for test_id in test_ids if test_id in all_ids]
    train_val_index = list(set(range(len(all_ids))) - set(test_index))
    assert len(train_val_index) == len(all_ids) - len(test_index)
    random.shuffle(train_val_index)
    if val_num is not None:
        n_val = val_num
    else:
        n_val = int(len(train_val_index) * val_ratio)
    val_index = train_val_index[:n_val]
    train_index = train_val_index[n_val:]
    return train_index, val_index, test_index


def time_split(index_path):
    valid_ids = np.loadtxt("./data/pdbbind_v2020/timesplit_no_lig_overlap_val", dtype=str)
    test_ids = np.loadtxt("./data/pdbbind_v2020/timesplit_test", dtype=str)

    with open(index_path, 'rb') as f:
        index = pickle.load(f)
    all_ids = [os.path.basename(i[0])[:4] for i in index]
    val_index = [all_ids.index(val_id) for val_id in valid_ids if val_id in all_ids]
    test_index = [all_ids.index(test_id) for test_id in test_ids if test_id in all_ids]
    train_index = list(set(range(len(all_ids))) - set(test_index) - set(val_index))
    return train_index, val_index, test_index


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--index_path', type=str, default='./data/pdbbind_v2016/pocket_10/index.pkl')
    parser.add_argument('--split_mode', type=str, choices=['coreset', 'time'], default='coreset')
    parser.add_argument('--test_path', type=str, default='./data/pdbbind_v2016/coreset')
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--val_num', type=int, default=None)
    parser.add_argument('--save_path', type=str, default='./data/pdbbind_v2016/pocket_10/split.pt')
    parser.add_argument('--seed', type=int, default=2021)
    args = parser.parse_args()
    random.seed(args.seed)

    if args.split_mode == 'coreset':
        train_index, val_index, test_index = coretest_split(args.index_path, args.test_path,
                                                            args.val_ratio, args.val_num)
    elif args.split_mode == 'time':
        train_index, val_index, test_index = time_split(args.index_path)
    else:
        raise ValueError(args.split_mode)
    torch.save({
        'train': train_index,
        'val': val_index,
        'test': test_index
    }, args.save_path)
    print('Train %d, Validation %d, Test %d.' % (len(train_index), len(val_index), len(test_index)))
    print('Done.')
