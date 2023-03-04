import argparse
import os
import pickle

import numpy as np
import torch
from torch_geometric.data import Batch
from torch_geometric.transforms import Compose
from tqdm.auto import tqdm

import utils.misc as misc
import utils.transforms as trans
from datasets import get_dataset
from datasets.pl_data import FOLLOW_BATCH
from models.molopt_score_model import ScorePosNet3D


def data_likelihood_estimation(model, data, time_steps, batch_size=1, device='cuda:0'):
    num_timesteps = len(time_steps)
    num_batch = int(np.ceil(num_timesteps / batch_size))
    all_kl_pos, all_kl_v = [], []

    cur_i = 0
    # t in [T-1, ..., 0]
    for i in range(num_batch):
        n_data = batch_size if i < num_batch - 1 else num_timesteps - batch_size * (num_batch - 1)
        batch = Batch.from_data_list([data.clone() for _ in range(n_data)], follow_batch=FOLLOW_BATCH).to(device)
        time_step = time_steps[cur_i:cur_i + n_data]

        kl_pos, kl_v = model.likelihood_estimation(
            protein_pos=batch.protein_pos,
            protein_v=batch.protein_atom_feature.float(),
            batch_protein=batch.protein_element_batch,

            ligand_pos=batch.ligand_pos,
            ligand_v=batch.ligand_atom_feature_full,
            batch_ligand=batch.ligand_element_batch,

            time_step=time_step
        )
        all_kl_pos.append(kl_pos)
        all_kl_v.append(kl_v)
        cur_i += n_data

    # prior
    batch = Batch.from_data_list([data.clone() for _ in range(1)], follow_batch=FOLLOW_BATCH).to(device)
    time_step = torch.tensor([model.num_timesteps], device=device)
    kl_pos_prior, kl_v_prior = model.likelihood_estimation(
        protein_pos=batch.protein_pos,
        protein_v=batch.protein_atom_feature.float(),
        batch_protein=batch.protein_element_batch,

        ligand_pos=batch.ligand_pos,
        ligand_v=batch.ligand_atom_feature_full,
        batch_ligand=batch.ligand_element_batch,

        time_step=time_step
    )
    all_kl_pos, all_kl_v = torch.cat(all_kl_pos), torch.cat(all_kl_v)
    sum_kl_pos, sum_kl_v = model.num_timesteps * torch.mean(all_kl_pos), model.num_timesteps * torch.mean(all_kl_v)
    all_kl_pos, all_kl_v = torch.cat([all_kl_pos, kl_pos_prior]), torch.cat([all_kl_v, kl_v_prior])
    sum_kl_pos += kl_pos_prior[0]
    sum_kl_v += kl_v_prior[0]
    return all_kl_pos.cpu(), all_kl_v.cpu(), sum_kl_pos.item(), sum_kl_v.item()


def get_dataset_result(dset, affinity_info):
    valid_id = []
    for data_id in tqdm(range(len(dset)), desc='Filtering data'):
        data = dset[data_id]
        ligand_fn_key = data.ligand_filename[:-4]
        pk = affinity_info[ligand_fn_key]['pk']
        if pk > 0:
            valid_id.append(data_id)
    print(f'There are {len(valid_id)} examples with valid pK in total.')

    all_results = []
    for data_id in tqdm(valid_id, desc='Evaluating'):
        data = dset[data_id]
        # likelihoods
        time_steps = torch.tensor(list(range(0, 1000, 100)), device=args.device)
        all_kl_pos, all_kl_v, sum_kl_pos, sum_kl_v = data_likelihood_estimation(
            model, data, time_steps, batch_size=args.batch_size, device=args.device)
        kl = sum_kl_pos + sum_kl_v

        # embedding
        batch = Batch.from_data_list([data.clone() for _ in range(1)], follow_batch=FOLLOW_BATCH).to(args.device)
        preds = model.fetch_embedding(
            protein_pos=batch.protein_pos,
            protein_v=batch.protein_atom_feature.float(),
            batch_protein=batch.protein_element_batch,

            ligand_pos=batch.ligand_pos,
            ligand_v=batch.ligand_atom_feature_full,
            batch_ligand=batch.ligand_element_batch,
        )

        # gather results
        ligand_fn_key = data.ligand_filename[:-4]
        result = {
            'idx': data_id,
            **affinity_info[ligand_fn_key],
            'kl_pos': all_kl_pos,
            'kl_v': all_kl_v,
            'nll': kl,
            'pred_ligand_v': preds['pred_ligand_v'].cpu(),
            'final_h': preds['final_h'].cpu(),
            'final_ligand_h': preds['final_ligand_h'].cpu()
        }
        all_results.append(result)
    return all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--config', type=str, default='configs/sampling/final_diffusion/aromatic_176k.yml')
    parser.add_argument('--affinity_path', type=str, default='affinity_info.pkl')
    parser.add_argument('--index_path', type=str, default='data/crossdocked_v1.1_rmsd1.0_pocket10/index.pkl')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--result_path', type=str, default='./outputs_embedding')

    args = parser.parse_args()

    logger = misc.get_logger('evaluate')

    if os.path.exists(args.affinity_path):
        with open(args.affinity_path, 'rb') as f:
            affinity_info = pickle.load(f)
    else:
        # collect index
        with open(args.index_path, 'rb') as f:
            index = pickle.load(f)
        affinity_info = {}
        for pdb_file, sdf_file, rmsd in index:
            affinity_info[sdf_file[:-4]] = {'rmsd': rmsd}

        # fetch reference vina score / binding affinity
        types_path = 'data/CrossDocked2020/types/it2_tt_v1.1_completeset_train0.types'
        with open(types_path, 'r') as f:
            for ln in tqdm(f.readlines()):
                # <label> <pK> <RMSD to crystal> <Receptor> <Ligand> # <Autodock Vina score>
                _, pk, rmsd, protein_fn, ligand_fn, vina = ln.split()
                ligand_raw_fn = ligand_fn[:ligand_fn.rfind('.')]
                if ligand_raw_fn in affinity_info:
                    affinity_info[ligand_raw_fn].update({
                        'pk': float(pk),
                        'vina': float(vina[1:])
                    })

        # save affinity info
        with open(args.affinity_path, 'wb') as f:
            pickle.dump(affinity_info, f)

    # Load config
    config = misc.load_config(args.config)
    logger.info(config)
    misc.seed_all(config.sample.seed)

    # Load checkpoint
    ckpt = torch.load(config.model.checkpoint, map_location=args.device)

    # Transforms
    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_featurizer = trans.FeaturizeLigandAtom(ckpt['config'].data.transform.ligand_atom_mode)
    transform_list = [
        protein_featurizer,
        ligand_featurizer,
        trans.FeaturizeLigandBond(),
    ]
    if ckpt['config'].data.transform.random_rot:
        transform_list.append(trans.RandomRotation())
    transform = Compose(transform_list)

    # Load dataset
    dataset, subsets = get_dataset(
        config=ckpt['config'].data,
        transform=transform
    )
    train_set, test_set = subsets['train'], subsets['test']
    logger.info(f'Successfully load the dataset (size: {len(test_set)})!')

    # Load model
    model = ScorePosNet3D(
        ckpt['config'].model,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim
    ).to(args.device)
    model.load_state_dict(ckpt['model'], strict=False if 'train_config' in config.model else True)
    logger.info(f'Successfully load the model! {config.model.checkpoint}')

    # filter data with valid pK, compute likelihood and embedding
    valid_train_results = get_dataset_result(train_set, affinity_info)
    valid_test_results = get_dataset_result(test_set, affinity_info)

    os.makedirs(args.result_path, exist_ok=True)
    torch.save(valid_train_results, os.path.join(args.result_path, 'crossdocked_train.pt'))
    torch.save(valid_test_results, os.path.join(args.result_path, 'crossdocked_test.pt'))
