import argparse
import os
import torch
from tqdm.auto import tqdm
from utils.evaluation.docking_qvina import QVinaDockingTask
from utils.evaluation.docking_vina import VinaDockingTask
import multiprocessing as mp


def dock_pocket_samples(pocket_samples):
    ligand_fn = pocket_samples[0]['ligand_filename']
    print('Start docking pocket: %s' % ligand_fn)
    pocket_results = []
    for idx, s in enumerate(tqdm(pocket_samples, desc='docking %d' % os.getpid())):
        try:
            if args.docking_mode == 'qvina':
                vina_task = QVinaDockingTask.from_generated_mol(
                    s['mol'], s['ligand_filename'], protein_root=args.protein_root, size_factor=args.dock_size_factor)
                vina_results = vina_task.run_sync()
            elif args.docking_mode == 'vina_score':
                vina_task = VinaDockingTask.from_generated_mol(
                    s['mol'], s['ligand_filename'], protein_root=args.protein_root)
                score_only_results = vina_task.run(mode='score_only', exhaustiveness=args.exhaustiveness)
                minimize_results = vina_task.run(mode='minimize', exhaustiveness=args.exhaustiveness)
                vina_results = {
                    'score_only': score_only_results,
                    'minimize': minimize_results
                }
            else:
                raise ValueError
        except:
            print('Error at %d of %s' % (idx, ligand_fn))
            vina_results = None
        pocket_results.append({**s, 'vina': vina_results})
    return pocket_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('sample_path', type=str)
    parser.add_argument('-o', '--out', type=str, default=None)
    parser.add_argument('-n', '--num_processes', type=int, default=10)
    parser.add_argument('--protein_root', type=str, default='./data/crossdocked_v1.1_rmsd1.0')
    parser.add_argument('--dock_size_factor', type=float, default=None)
    parser.add_argument('--exhaustiveness', type=int, default=16)
    parser.add_argument('--docking_mode', type=str, default='vina_score',
                        choices=['none', 'qvina', 'vina_score'])
    args = parser.parse_args()

    samples = torch.load(args.sample_path)
    with mp.Pool(args.num_processes) as p:
        docked_samples = p.map(dock_pocket_samples, samples)
    if args.out is None:
        dir_name = os.path.dirname(args.sample_path)
        baseline_name = os.path.basename(args.sample_path).split('_')[0]
        out_path = os.path.join(dir_name, baseline_name + '_test_docked.pt')
    else:
        out_path = args.out
    torch.save(docked_samples, out_path)
