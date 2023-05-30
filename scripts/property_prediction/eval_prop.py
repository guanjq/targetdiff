import argparse
from tqdm.auto import tqdm
import torch
import torch.utils.tensorboard
from torch_geometric.transforms import Compose

from datasets import get_dataset
import utils.transforms_prop as utils_trans
import utils.misc as utils_misc
from utils.misc_prop import get_model, get_dataloader, get_eval_scores
import numpy as np
from datasets.protein_ligand import KMAP


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str)
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
    # logger.info(f'# trainable parameters: {utils_misc.count_parameters(model) / 1e6:.4f} M')
    model.eval()

    # Datasets and loaders
    # config.dataset.path = './data/pdbbind_v2020/pocket_10_refined'
    # config.dataset.split = './data/pdbbind_v2020/pocket_10_refined/split.pt'
    logger.info('Loading dataset...')
    dataset, subsets = get_dataset(
        config=config.dataset,
        transform=transform,
        heavy_only=config.dataset.get('heavy_only', False)
    )
    train_set, val_set, test_set = subsets['train'], subsets['val'], subsets['test']
    logger.info(f'Train set: {len(train_set)} Val set: {len(val_set)} Test set: {len(test_set)}')
    train_loader, val_loader, test_loader = get_dataloader(train_set, val_set, test_set, config)

    def validate(epoch, data_loader, prefix='Test'):
        sum_loss, sum_n = 0, 0
        ytrue_arr, ypred_arr = [], []
        y_kind = []
        with torch.no_grad():
            model.eval()
            for batch in tqdm(data_loader, desc=prefix):
                batch = batch.to(args.device)
                loss, pred = model.get_loss(batch, pos_noise_std=0., return_pred=True)
                sum_loss += loss.item() * len(batch.y)
                sum_n += len(batch.y)
                ypred_arr.append(pred.view(-1))
                ytrue_arr.append(batch.y)
                y_kind.append(batch.kind)
        avg_loss = sum_loss / sum_n
        logger.info('[%s] Epoch %03d | Loss %.6f' % (
            prefix, epoch, avg_loss,
        ))
        ypred_arr = torch.cat(ypred_arr).cpu().numpy().astype(np.float64)
        ytrue_arr = torch.cat(ytrue_arr).cpu().numpy().astype(np.float64)
        y_kind = torch.cat(y_kind).cpu().numpy()
        rmse = get_eval_scores(ypred_arr, ytrue_arr, logger)
        for k, v in KMAP.items():
            get_eval_scores(ypred_arr[y_kind == v], ytrue_arr[y_kind == v], logger, prefix=k)
        return avg_loss

    test_loss = validate(ckpt_restore['epoch'], test_loader)
    print('Test loss: ', test_loss)


if __name__ == '__main__':
    main()
