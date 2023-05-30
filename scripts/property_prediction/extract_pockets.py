import argparse
import multiprocessing as mp
import os
import pickle
from functools import partial

from rdkit import RDLogger
from tqdm.auto import tqdm

from datasets.protein_ligand import parse_sdf_file_mol, read_mol, KMAP
from utils.data import PDBProtein

RDLogger.DisableLog('rdApp.*')


def parse_pdbbind_index_file(raw_path, subset='refined'):
    all_index = []
    version = int(raw_path.rstrip('/')[-4:])
    assert version >= 2016
    if subset == 'refined':
        data_path = os.path.join(raw_path, f'refined-set')
        index_path = os.path.join(data_path, 'index', f'INDEX_refined_data.{version}')
    elif subset == 'general':
        data_path = os.path.join(raw_path, f'general-set-except-refined')
        index_path = os.path.join(data_path, 'index', f'INDEX_general_PL_data.{version}')
    else:
        raise ValueError(subset)

    all_files = os.listdir(data_path)
    with open(index_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if line.startswith('#'): continue
        index, res, year, pka, kv = line.split('//')[0].strip().split()
        kind = [v for k, v in KMAP.items() if k in kv]
        assert len(kind) == 1
        if index in all_files:
            all_index.append([index, res, year, pka, kind[0]])
    return all_index


def process_item(item, args):
    pdb_idx, res, year, pka, kind = item
    try:
        if args.subset == 'refined':
            pdb_path = os.path.join(args.source, 'refined-set', pdb_idx)
        elif args.subset == 'general':
            pdb_path = os.path.join(args.source, 'general-set-except-refined', pdb_idx)
        else:
            raise ValueError(args.subset)

        protein_path = os.path.join(pdb_path, f'{pdb_idx}_protein.pdb')
        ligand_sdf_path = os.path.join(pdb_path, f'{pdb_idx}_ligand.sdf')
        ligand_mol2_path = os.path.join(pdb_path, f'{pdb_idx}_ligand.mol2')
        mol, problem, ligand_path = read_mol(ligand_sdf_path, ligand_mol2_path)
        if problem:
            print('Read mol error.', item)
            return None, ligand_path, res, pka, kind

        protein = PDBProtein(protein_path)
        # ligand = parse_sdf_file_mol(ligand_path, heavy_only=True)
        ligand = parse_sdf_file_mol(ligand_path, heavy_only=False)
        pocket_path = os.path.join(pdb_path, f'{pdb_idx}_pocket{args.radius}.pdb')
        if not os.path.exists(pocket_path):
            pdb_block_pocket = protein.residues_to_pdb_block(
                protein.query_residues_ligand(ligand, args.radius)
            )
            with open(pocket_path, 'w') as f:
                f.write(pdb_block_pocket)
        return pocket_path, ligand_path, res, pka, kind

    except Exception:
        print('Exception occured.', item)
        return None, ligand_path, res, pka, kind


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='./data/pdbbind_v2016')
    parser.add_argument('--fixed_sdf_dir', type=str, default='./data/pdbbind_v2016/fixed_sdf_files')
    parser.add_argument('--subset', type=str, default='refined')
    parser.add_argument('--refined_index_pkl', type=str, default=None)
    parser.add_argument('--radius', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=16)
    args = parser.parse_args()

    index = parse_pdbbind_index_file(args.source, args.subset)
    # if not os.path.exists(args.fixed_sdf_dir):
    #     os.makedirs(args.fixed_sdf_dir)

    pool = mp.Pool(args.num_workers)
    index_pocket = []
    for item_pocket in tqdm(pool.imap_unordered(partial(process_item, args=args), index), total=len(index)):
        index_pocket.append(item_pocket)
    pool.close()

    valid_index_pocket = []
    for index in index_pocket:
        if index[0] is not None:
            valid_index_pocket.append(index)

    save_path = os.path.join(args.source, f'pocket_{args.radius}_{args.subset}')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    index_path = os.path.join(save_path, 'index.pkl')

    if args.subset == 'general' and args.refined_index_pkl is not None:
        with open(args.refined_index_pkl, 'rb') as f:
            refined_index = pickle.load(f)
        valid_index_pocket += refined_index
    with open(index_path, 'wb') as f:
        pickle.dump(valid_index_pocket, f)
    print('Done. %d protein-ligand pairs in total.' % len(valid_index_pocket))
