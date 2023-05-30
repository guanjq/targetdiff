import argparse

import torch
import torch.utils.tensorboard
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm
from torch_geometric.transforms import Compose

import utils.misc as utils_misc
from datasets.protein_ligand import KMAP, parse_sdf_file_mol
from datasets.pl_data import ProteinLigandData, torchify_dict
from utils.data import PDBProtein
import utils.transforms_prop as utils_trans
from utils.misc_prop import get_model


class InferenceDataset(Dataset):
    def __init__(self, data_list):
        super().__init__()
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        return data


def convert_data(pdb_path, ligand_path, transform, radius=10, pocket=False, heavy_only=False):
    ligand_dict = parse_sdf_file_mol(ligand_path, heavy_only=heavy_only)
    if not pocket:
        protein = PDBProtein(pdb_path)
        pdb_block_pocket = protein.residues_to_pdb_block(
            protein.query_residues_ligand(ligand_dict, radius)
        )
        pocket_dict = PDBProtein(pdb_block_pocket).to_dict_atom()
    else:
        pocket_dict = PDBProtein(pdb_path).to_dict_atom()

    data = ProteinLigandData.from_protein_ligand_dicts(
        protein_dict=torchify_dict(pocket_dict),
        ligand_dict=torchify_dict(ligand_dict),
    )
    data.protein_filename = pdb_path
    data.ligand_filename = ligand_path
    assert data.protein_pos.size(0) > 0
    if transform is not None:
        data = transform(data)
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str)
    parser.add_argument('--protein_path', type=str)
    parser.add_argument('--ligand_path', type=str)
    parser.add_argument('--kind', type=str, default='Ki', choices=['Ki', 'Kd', 'IC50'])
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=2021)
    args = parser.parse_args()
    utils_misc.seed_all(args.seed)

    # Logging
    logger = utils_misc.get_logger('eval')
    logger.info(args)

    # Load config
    logger.info(f'Loading model from {args.ckpt_path}')
    ckpt_restore = torch.load(args.ckpt_path, map_location=torch.device('cpu'))
    config = ckpt_restore['config']
    logger.info(f'ckpt_config: {config}')

    # Transforms
    protein_featurizer = utils_trans.FeaturizeProteinAtom()
    ligand_featurizer = utils_trans.FeaturizeLigandAtom()
    transform = Compose([
        protein_featurizer,
        ligand_featurizer,
    ])

    # Load model
    model = get_model(config, protein_featurizer.feature_dim, ligand_featurizer.feature_dim)
    model.load_state_dict(ckpt_restore['model'])
    model = model.to(args.device)
    # print(model)
    logger.info(f'# trainable parameters: {utils_misc.count_parameters(model) / 1e6:.4f} M')
    model.eval()

    test_data = convert_data(args.protein_path, args.ligand_path, transform,
                             heavy_only=config.dataset.get('heavy_only', False))
    test_data.kind = KMAP[args.kind]
    test_set = InferenceDataset([test_data])
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False,
                             follow_batch=['protein_element', 'ligand_element'])

    with torch.no_grad():
        model.eval()
        for batch in tqdm(test_loader, desc='Inference'):
            batch = batch.to(args.device)
            pred = model(
                protein_pos=batch.protein_pos,
                protein_atom_feature=batch.protein_atom_feature.float(),
                ligand_pos=batch.ligand_pos,
                ligand_atom_feature=batch.ligand_atom_feature_full.float(),
                batch_protein=batch.protein_element_batch,
                batch_ligand=batch.ligand_element_batch,
                output_kind=batch.kind
            )

            print(f'PDB ID: {batch.protein_filename[0]} '
                  f'Prediction: {args.kind}={unit_transform(pred.cpu().squeeze()):.2e} m')


def unit_transform(pka):
    # pka = -log10 Kd / Ki
    affinity = torch.pow(10, -pka.cpu().squeeze())
    return affinity


if __name__ == '__main__':
    main()
