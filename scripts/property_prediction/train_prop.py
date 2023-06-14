import os
import shutil
import argparse
from tqdm.auto import tqdm
import torch
from torch.nn.utils import clip_grad_norm_
import torch.utils.tensorboard
from torch_geometric.transforms import Compose

from datasets import get_dataset
import utils.transforms_prop as utils_trans
import utils.misc as utils_misc
from utils.misc_prop import get_model, get_dataloader, get_eval_scores
from utils.train import get_scheduler, get_optimizer
import numpy as np
from datasets.protein_ligand import KMAP


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--logdir', type=str, default='./logs')
    parser.add_argument('--tag', type=str, default='')
    args = parser.parse_args()

    # Load configs
    config = utils_misc.load_config(args.config)
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    utils_misc.seed_all(config.train.seed)

    # Logging
    log_dir = utils_misc.get_new_log_dir(args.logdir, prefix=config_name, tag=args.tag)
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = utils_misc.get_logger('train', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    logger.info(args)
    logger.info(config)
    shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))
    shutil.copytree('./models', os.path.join(log_dir, 'models'))

    # Transforms
    protein_featurizer = utils_trans.FeaturizeProteinAtom()
    ligand_featurizer = utils_trans.FeaturizeLigandAtom()
    transform = Compose([
        protein_featurizer,
        ligand_featurizer,
    ])

    # Datasets and loaders
    logger.info('Loading dataset...')
    dataset, subsets = get_dataset(
        config=config.dataset,
        transform=transform,
        emb_path=config.dataset.emb_path if 'emb_path' in config.dataset else None,
        heavy_only=config.dataset.heavy_only
    )
    train_set, val_set, test_set = subsets['train'], subsets['val'], subsets['test']
    logger.info(f'Train set: {len(train_set)} Val set: {len(val_set)} Test set: {len(test_set)}')
    train_loader, val_loader, test_loader = get_dataloader(train_set, val_set, test_set, config)
    # Model
    logger.info('Building model...')
    model = get_model(config, protein_featurizer.feature_dim, ligand_featurizer.feature_dim)
    model = model.to(args.device)
    logger.info(f'# trainable parameters: {utils_misc.count_parameters(model) / 1e6:.4f} M')

    # Optimizer and scheduler
    optimizer = get_optimizer(config.train.optimizer, model)
    scheduler = get_scheduler(config.train.scheduler, optimizer)

    def train(epoch):
        model.train()
        optimizer.zero_grad()
        it = 0
        num_it = len(train_loader)
        for batch in tqdm(train_loader, dynamic_ncols=True, desc=f'Epoch {epoch}', position=1):
            it += 1
            batch = batch.to(args.device)
            # compute loss
            loss = model.get_loss(batch, pos_noise_std=config.train.pos_noise_std)
            loss.backward()
            orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

            if it % config.train.report_iter == 0:
                logger.info('[Train] Epoch %03d Iter %04d | Loss %.6f | Lr %.4f * 1e-3' % (
                    epoch, it, loss.item(), optimizer.param_groups[0]['lr'] * 1000
                ))

            writer.add_scalar('train/loss', loss, it + epoch * num_it)
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it + epoch * num_it)
            writer.add_scalar('train/grad', orig_grad_norm, it + epoch * num_it)
            writer.flush()

    def validate(epoch, data_loader, scheduler, writer, prefix='Validate'):
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

        if scheduler:
            if config.train.scheduler.type == 'plateau':
                scheduler.step(avg_loss)
            elif config.train.scheduler.type == 'warmup_plateau':
                scheduler.step_ReduceLROnPlateau(avg_loss)
            else:
                scheduler.step()

        if writer:
            writer.add_scalar('val/loss', avg_loss, epoch)
            writer.add_scalar('val/rmse', rmse, epoch)
            writer.flush()

        return avg_loss

    try:
        best_val_loss = float('inf')
        best_val_epoch = 0
        patience = 0
        for epoch in range(1, config.train.max_epochs + 1):
            # with torch.autograd.detect_anomaly():
            train(epoch)
            if epoch % config.train.val_freq == 0 or epoch == config.train.max_epochs:
                val_loss = validate(epoch, val_loader, scheduler, writer)
                validate(epoch, test_loader, scheduler=None, writer=None, prefix='Test')

                if val_loss < best_val_loss:
                    patience = 0
                    best_val_loss = val_loss
                    best_val_epoch = epoch
                    logger.info(f'Best val achieved at epoch {epoch}, val loss: {best_val_loss:.3f}')
                    logger.info(f'Eval on Test set:')
                    validate(epoch, test_loader, scheduler=None, writer=None, prefix='Test')
                    ckpt_path = os.path.join(ckpt_dir, '%d.pt' % epoch)
                    torch.save({
                        'config': config,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'epoch': epoch,
                    }, ckpt_path)
                    logger.info(f'Model {log_dir}/{epoch}.pt saved!')
                else:
                    patience += 1
                    logger.info(f'Val loss does not improve, patience: {patience} '
                                f'(Best val loss: {best_val_loss:.3f} at epoch {best_val_epoch})')

    except KeyboardInterrupt:
        logger.info('Terminating...')


if __name__ == '__main__':
    main()
