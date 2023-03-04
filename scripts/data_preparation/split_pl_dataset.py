import os
import argparse
import random
import torch
from tqdm.auto import tqdm

from torch.utils.data import Subset
from datasets.pl_pair_dataset import PocketLigandPairDataset


def get_chain_name(fn):
    return os.path.basename(fn)[:6]


def get_pdb_name(fn):
    return os.path.basename(fn)[:4]


def get_unique_pockets(dataset, raw_id, used_pdb, num_pockets):
    # only save first encountered id for unseen pdbs
    unique_id = []
    pdb_visited = set()
    for idx in tqdm(raw_id, 'Filter'):
        pdb_name = get_pdb_name(dataset[idx].ligand_filename)
        if pdb_name not in used_pdb and pdb_name not in pdb_visited:
            unique_id.append(idx)
            pdb_visited.add(pdb_name)

    print('Number of Pairs: %d' % len(unique_id))
    print('Number of PDBs:  %d' % len(pdb_visited))

    random.Random(args.seed).shuffle(unique_id)
    unique_id = unique_id[:num_pockets]
    print('Number of selected: %d' % len(unique_id))
    return unique_id, pdb_visited.union(used_pdb)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='./data/crossdocked_v1.1_rmsd1.0_pocket10')
    parser.add_argument('--dest', type=str, default='./data/crossdocked_pocket10_pose_split.pt')
    parser.add_argument('--fixed_split', type=str, default='./data/split_by_name.pt')
    parser.add_argument('--train', type=int, default=100000)
    parser.add_argument('--val', type=int, default=1000)
    parser.add_argument('--test', type=int, default=20000)
    parser.add_argument('--val_num_pockets', type=int, default=-1)
    parser.add_argument('--test_num_pockets', type=int, default=100)
    parser.add_argument('--seed', type=int, default=2021)
    args = parser.parse_args()

    dataset = PocketLigandPairDataset(args.path)
    print('Load dataset successfully!')
    if args.fixed_split:
        fixed_split = torch.load(args.fixed_split)
        print('Load fixed split successfully!')
        name_id_dict = {}
        for idx, data in enumerate(tqdm(dataset, desc='Indexing')):
            name_id_dict[data.protein_filename + data.ligand_filename] = idx

        selected_ids = {'train': [], 'test': []}
        for split in ['train', 'test']:
            print(f'Selecting {split} split...')
            for fn in fixed_split[split]:
                if (fn[0] + fn[1]) in name_id_dict:
                    selected_ids[split].append(name_id_dict[fn[0] + fn[1]])
                else:
                    print(f'Warning: data with PDB fn {fn[0]} and ligand fn {fn[1]} not found!')
        train_id, val_id, test_id = selected_ids['train'], [], selected_ids['test']

    else:
        allowed_elements = {1, 6, 7, 8, 9, 15, 16, 17}
        elements = {i: set() for i in range(90)}
        for i, data in enumerate(tqdm(dataset, desc='Filter')):
            for e in data.ligand_element:
                elements[e.item()].add(i)

        all_id = set(range(len(dataset)))
        blocked_id = set().union(*[
            elements[i] for i in elements.keys() if i not in allowed_elements
        ])

        allowed_id = list(all_id - blocked_id)
        random.Random(args.seed).shuffle(allowed_id)
        print('Allowed: %d' % len(allowed_id))

        train_id = allowed_id[:args.train]
        train_set = Subset(dataset, indices=train_id)
        train_pdb = {get_pdb_name(d.ligand_filename) for d in tqdm(train_set)}
        print('train pdb: ', train_pdb)

        if args.val_num_pockets == -1:
            # not group by pocket
            val_id = allowed_id[args.train: args.train + args.val]
            used_pdb = train_pdb
        else:
            raw_val_id = allowed_id[args.train: args.train + args.val]
            val_id, used_pdb = get_unique_pockets(dataset, raw_val_id, train_pdb, args.val_num_pockets)

        if args.test_num_pockets == -1:
            test_id = allowed_id[args.train + args.val: args.train + args.val + args.test]
        else:
            raw_test_id = allowed_id[args.train + args.val: args.train + args.val + args.test]
            test_id, used_pdb = get_unique_pockets(dataset, raw_test_id, used_pdb, args.test_num_pockets)

    torch.save({
        'train': train_id,
        'val': val_id,
        'test': test_id,
    }, args.dest)

    print('Train %d, Validation %d, Test %d.' % (len(train_id), len(val_id), len(test_id)))
    print('Done.')
