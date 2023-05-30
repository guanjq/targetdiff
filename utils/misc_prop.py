import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch_geometric.loader import DataLoader

from models.property_pred.prop_model import PropPredNet, PropPredNetEnc


def get_eval_scores(ypred_arr, ytrue_arr, logger, prefix='All'):
    if len(ypred_arr) == 0:
        return None
    rmse = np.sqrt(mean_squared_error(ytrue_arr, ypred_arr))
    mae = mean_absolute_error(ytrue_arr, ypred_arr)
    r2 = r2_score(ytrue_arr, ypred_arr)
    pearson, ppval = pearsonr(ytrue_arr, ypred_arr)
    spearman, spval = spearmanr(ytrue_arr, ypred_arr)
    mean = np.mean(ypred_arr)
    std = np.std(ypred_arr)
    logger.info("Evaluation Summary:")
    logger.info(
        "[%4s] num: %3d, RMSE: %.3f, MAE: %.3f, "
        "R^2 score: %.3f, Pearson: %.3f, Spearman: %.3f, mean/std: %.3f/%.3f" % (
            prefix, len(ypred_arr), rmse, mae, r2, pearson, spearman, mean, std))
    return rmse


def get_dataloader(train_set, val_set, test_set, config):
    follow_batch = ['protein_element', 'ligand_element']
    collate_exclude_keys = ['ligand_nbh_list']
    train_loader = DataLoader(
        train_set,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
        follow_batch=follow_batch,
        exclude_keys=collate_exclude_keys
    )
    val_loader = DataLoader(val_set, config.train.batch_size, shuffle=False,
                            follow_batch=follow_batch, exclude_keys=collate_exclude_keys)
    test_loader = DataLoader(test_set, config.train.batch_size, shuffle=False,
                             follow_batch=follow_batch, exclude_keys=collate_exclude_keys)
    return train_loader, val_loader, test_loader


def get_model(config, protein_atom_feat_dim, ligand_atom_feat_dim):
    if config.model.encoder.name == 'egnn_enc':
        model = PropPredNetEnc(
            config.model,
            protein_atom_feature_dim=protein_atom_feat_dim,
            ligand_atom_feature_dim=ligand_atom_feat_dim,
            enc_ligand_dim=config.model.enc_ligand_dim,
            enc_node_dim=config.model.enc_node_dim,
            enc_graph_dim=config.model.enc_graph_dim,
            enc_feature_type=config.model.enc_feature_type,
            output_dim=1
        )
    else:
        model = PropPredNet(
            config.model,
            protein_atom_feature_dim=protein_atom_feat_dim,
            ligand_atom_feature_dim=ligand_atom_feat_dim,
            output_dim=3
        )
    return model
