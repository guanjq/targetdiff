import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter

from models.property_pred.prop_egnn import EnEquiEncoder
from models.common import compose_context_prop, ShiftedSoftplus


def get_encoder(config):
    if config.name == 'egnn' or config.name == 'egnn_enc':
        net = EnEquiEncoder(
            num_layers=config.num_layers,
            edge_feat_dim=config.edge_dim,
            hidden_dim=config.hidden_dim,
            num_r_gaussian=config.num_r_gaussian,
            act_fn=config.act_fn,
            norm=config.norm,
            update_x=False,
            k=config.knn,
            cutoff=config.cutoff,
        )
    else:
        raise ValueError(config.name)
    return net


class PropPredNet(nn.Module):
    def __init__(self, config, protein_atom_feature_dim, ligand_atom_feature_dim, output_dim=3):
        super(PropPredNet, self).__init__()
        self.config = config
        self.hidden_dim = config.hidden_channels
        self.output_dim = output_dim
        self.protein_atom_emb = nn.Linear(protein_atom_feature_dim, self.hidden_dim)
        self.ligand_atom_emb = nn.Linear(ligand_atom_feature_dim, self.hidden_dim)

        # self.mean = target_mean
        # self.std = target_std
        # self.register_buffer('target_mean', target_mean)
        # self.register_buffer('target_std', target_std)
        self.encoder = get_encoder(config.encoder)
        self.out_block = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            ShiftedSoftplus(),
            nn.Linear(self.hidden_dim, output_dim),
        )

    def forward(self, protein_pos, protein_atom_feature, ligand_pos, ligand_atom_feature, batch_protein, batch_ligand,
                output_kind):
        h_protein = self.protein_atom_emb(protein_atom_feature)
        h_ligand = self.ligand_atom_emb(ligand_atom_feature)

        h_ctx, pos_ctx, batch_ctx = compose_context_prop(
            h_protein=h_protein,
            h_ligand=h_ligand,
            pos_protein=protein_pos,
            pos_ligand=ligand_pos,
            batch_protein=batch_protein,
            batch_ligand=batch_ligand,
        )

        h_ctx = self.encoder(
            node_attr=h_ctx,
            pos=pos_ctx,
            batch=batch_ctx,
        )  # (N_p+N_l, H)

        # Aggregate messages
        pre_out = scatter(h_ctx, index=batch_ctx, dim=0, reduce='sum')  # (N, F)
        output = self.out_block(pre_out)  # (N, C)
        if output_kind is not None:
            output_mask = F.one_hot(output_kind - 1, self.output_dim)
            output = torch.sum(output * output_mask, dim=-1, keepdim=True)
        return output

    def get_loss(self, batch, pos_noise_std, return_pred=False):
        protein_noise = torch.randn_like(batch.protein_pos) * pos_noise_std
        ligand_noise = torch.randn_like(batch.ligand_pos) * pos_noise_std
        pred = self(
            protein_pos=batch.protein_pos + protein_noise,
            protein_atom_feature=batch.protein_atom_feature.float(),
            ligand_pos=batch.ligand_pos + ligand_noise,
            ligand_atom_feature=batch.ligand_atom_feature_full.float(),
            batch_protein=batch.protein_element_batch,
            batch_ligand=batch.ligand_element_batch,
            output_kind=batch.kind,
            # output_kind=None
        )
        # pred = pred * y_std + y_mean
        loss_func = nn.MSELoss()
        loss = loss_func(pred.view(-1), batch.y)
        if return_pred:
            return loss, pred
        else:
            return loss


class PropPredNetEnc(nn.Module):
    def __init__(self, config, protein_atom_feature_dim, ligand_atom_feature_dim,
                 enc_ligand_dim, enc_node_dim, enc_graph_dim, enc_feature_type=None, output_dim=1):
        super(PropPredNetEnc, self).__init__()
        self.config = config
        self.hidden_dim = config.hidden_channels
        self.output_dim = output_dim
        self.enc_ligand_dim = enc_ligand_dim
        self.enc_node_dim = enc_node_dim
        self.enc_graph_dim = enc_graph_dim
        self.enc_feature_type = enc_feature_type

        self.protein_atom_emb = nn.Linear(protein_atom_feature_dim, self.hidden_dim)
        self.ligand_atom_emb = nn.Linear(ligand_atom_feature_dim + enc_ligand_dim, self.hidden_dim)
        # self.mean = target_mean
        # self.std = target_std
        # self.register_buffer('target_mean', target_mean)
        # self.register_buffer('target_std', target_std)
        self.encoder = get_encoder(config.encoder)
        if self.enc_node_dim > 0:
            self.enc_node_layer = nn.Sequential(
                nn.Linear(self.hidden_dim + self.enc_node_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )

        self.out_block = nn.Sequential(
            nn.Linear(self.hidden_dim + self.enc_graph_dim, self.hidden_dim),
            ShiftedSoftplus(),
            nn.Linear(self.hidden_dim, output_dim),
        )

    def forward(self, protein_pos, protein_atom_feature, ligand_pos, ligand_atom_feature, batch_protein, batch_ligand,
                output_kind, enc_ligand_feature, enc_node_feature, enc_graph_feature):
        h_protein = self.protein_atom_emb(protein_atom_feature)
        if enc_ligand_feature is not None:
            ligand_atom_feature = torch.cat([ligand_atom_feature, enc_ligand_feature], dim=-1)
        h_ligand = self.ligand_atom_emb(ligand_atom_feature)

        h_ctx, pos_ctx, batch_ctx = compose_context_prop(
            h_protein=h_protein,
            h_ligand=h_ligand,
            pos_protein=protein_pos,
            pos_ligand=ligand_pos,
            batch_protein=batch_protein,
            batch_ligand=batch_ligand,
        )

        h_ctx = self.encoder(
            node_attr=h_ctx,
            pos=pos_ctx,
            batch=batch_ctx,
        )  # (N_p+N_l, H)

        if enc_node_feature is not None:
            h_ctx = torch.cat([h_ctx, enc_node_feature], dim=-1)
            h_ctx = self.enc_node_layer(h_ctx)

        # Aggregate messages
        pre_out = scatter(h_ctx, index=batch_ctx, dim=0, reduce='sum')  # (N, F)
        if enc_graph_feature is not None:
            pre_out = torch.cat([pre_out, enc_graph_feature], dim=-1)

        output = self.out_block(pre_out)  # (N, C)
        if output_kind is not None:
            output_mask = F.one_hot(output_kind - 1, self.output_dim)
            output = torch.sum(output * output_mask, dim=-1, keepdim=True)
        return output

    def get_loss(self, batch, pos_noise_std, return_pred=False):
        protein_noise = torch.randn_like(batch.protein_pos) * pos_noise_std
        ligand_noise = torch.randn_like(batch.ligand_pos) * pos_noise_std

        # add features
        enc_ligand_feature, enc_node_feature, enc_graph_feature = None, None, None
        if self.enc_feature_type == 'nll_all':
            enc_graph_feature = batch.nll_all  # [num_graphs, 22]
        elif self.enc_feature_type == 'nll':
            enc_graph_feature = batch.nll  # [num_graphs, 20]
        elif self.enc_feature_type == 'final_h':
            enc_node_feature = batch.final_h  # [num_pl_atoms, 128]
        elif self.enc_feature_type == 'pred_ligand_v':
            enc_ligand_feature = batch.pred_ligand_v  # [num_l_atoms, 13]
        elif self.enc_feature_type == 'pred_v_entropy_pre':
            enc_ligand_feature = batch.pred_v_entropy   # [num_l_atoms, 1]
        elif self.enc_feature_type == 'pred_v_entropy_post':
            enc_graph_feature = scatter(batch.pred_v_entropy, index=batch.ligand_element_batch, dim=0, reduce='sum')   # [num_graphs, 1]
        elif self.enc_feature_type == 'full':
            enc_graph_feature = torch.cat(
                [batch.nll_all, scatter(batch.pred_v_entropy, index=batch.ligand_element_batch, dim=0, reduce='sum')], dim=-1)
            enc_node_feature = batch.final_h
            enc_ligand_feature = torch.cat([batch.pred_ligand_v, batch.pred_v_entropy], -1)
        else:
            raise NotImplementedError

        pred = self(
            protein_pos=batch.protein_pos + protein_noise,
            protein_atom_feature=batch.protein_atom_feature.float(),
            ligand_pos=batch.ligand_pos + ligand_noise,
            ligand_atom_feature=batch.ligand_atom_feature_full.float(),
            batch_protein=batch.protein_element_batch,
            batch_ligand=batch.ligand_element_batch,
            output_kind=batch.kind,
            # output_kind=None,
            enc_ligand_feature=enc_ligand_feature,
            enc_node_feature=enc_node_feature,
            enc_graph_feature=enc_graph_feature
        )
        # pred = pred * y_std + y_mean
        loss_func = nn.MSELoss()
        loss = loss_func(pred.view(-1), batch.y)
        if return_pred:
            return loss, pred
        else:
            return loss
