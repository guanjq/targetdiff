import argparse
import os

from rdkit import Chem
import torch
from tqdm.auto import tqdm
from utils import misc
from utils.evaluation import scoring_func
from utils.evaluation.docking_qvina import QVinaDockingTask
from datasets import get_dataset
from easydict import EasyDict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='./data/crossdocked_v1.1_rmsd1.0_pocket10')
    parser.add_argument('-s', '--split', type=str, default='./data/crossdocked_pocket10_pose_split.pt')
    parser.add_argument('-o', '--out', type=str, default=None)
    parser.add_argument('--protein_root', type=str, default='./data/crossdocked_v1.1_rmsd1.0')
    parser.add_argument('--ligand_root', type=str, default='./data/crossdocked_v1.1_rmsd1.0_pocket10')
    parser.add_argument('--use_uff', type=eval, default=True)
    parser.add_argument('--size_factor', type=float, default=1.2)
    args = parser.parse_args()

    logger = misc.get_logger('docking')
    logger.info(args)

    # Load dataset
    dataset, subsets = get_dataset(
        config=EasyDict({
            'name': 'pl',
            'path': args.dataset,
            'split': args.split
        })
    )
    train_set, test_set = subsets['train'], subsets['test']
    logger.info(f'Successfully load the dataset (size: {len(test_set)})!')

    # Dock
    logger.info('Start docking...')
    results = []
    for i, data in enumerate(tqdm(test_set)):
        mol = next(iter(Chem.SDMolSupplier(os.path.join(args.ligand_root, data.ligand_filename))))
        # try:
        chem_results = scoring_func.get_chem(mol)
        vina_task = QVinaDockingTask.from_original_data(
            data,
            ligand_root=args.ligand_root,
            protein_root=args.protein_root,
            use_uff=args.use_uff,
            size_factor=args.size_factor
        )
        vina_results = vina_task.run_sync()
        # except:
        #     logger.warning('Error #%d' % i)
        #     continue

        results.append({
            'mol': mol,
            'smiles': data.ligand_smiles,
            'ligand_filename': data.ligand_filename,
            'chem_results': chem_results,
            'vina': vina_results
        })

    # Save
    if args.out is None:
        split_name = os.path.basename(args.split)
        split_name = split_name[:split_name.rfind('.')]
        docked_name = f'{split_name}_test_docked_uff_{args.use_uff}_size_{args.size_factor}.pt'
        out_path = os.path.join(os.path.dirname(args.dataset), docked_name)
    else:
        out_path = args.out
    logger.info('Num docked: %d' % len(results))
    logger.info('Saving results to %s' % out_path)
    torch.save(results, out_path)
