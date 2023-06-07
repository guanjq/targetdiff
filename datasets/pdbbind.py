import os
import pickle
import lmdb
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from utils.data import PDBProtein
from datasets.protein_ligand import parse_sdf_file_mol
from datasets.pl_data import ProteinLigandData, torchify_dict
from scipy import stats


class PDBBindDataset(Dataset):

    def __init__(self, raw_path, transform=None, emb_path=None, heavy_only=False):
        super().__init__()
        self.raw_path = raw_path.rstrip('/')
        self.index_path = os.path.join(self.raw_path, 'index.pkl')
        self.processed_path = os.path.join(self.raw_path, os.path.basename(self.raw_path) + '_processed.lmdb')
        self.emb_path = emb_path
        self.transform = transform
        self.heavy_only = heavy_only
        self.db = None

        self.keys = None

        if not os.path.exists(self.processed_path):
            self._process()
        print('Load dataset from ', self.processed_path)
        if self.emb_path is not None:
            print('Load embedding from ', self.emb_path)
            self.emb = torch.load(self.emb_path)

    def _connect_db(self):
        """
            Establish read-only database connection
        """
        assert self.db is None, 'A connection has already been opened.'
        self.db = lmdb.open(
            self.processed_path,
            map_size=10*(1024*1024*1024),   # 10GB
            create=False,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.db.begin() as txn:
            self.keys = list(txn.cursor().iternext(values=False))

    def _close_db(self):
        self.db.close()
        self.db = None
        self.keys = None
        
    def _process(self):
        db = lmdb.open(
            self.processed_path,
            map_size=10*(1024*1024*1024),   # 10GB
            create=True,
            subdir=False,
            readonly=False,  # Writable
        )
        with open(self.index_path, 'rb') as f:
            index = pickle.load(f)

        # index = parse_pdbbind_index_file(self.index_path)

        num_skipped = 0
        with db.begin(write=True, buffers=True) as txn:
            for i, (pocket_fn, ligand_fn, resolution, pka, kind) in enumerate(tqdm(index)):
                # try:
                # pdb_path = os.path.join(self.raw_path, 'refined-set', pdb_idx)
                # pocket_fn = os.path.join(pdb_path, f'{pdb_idx}_pocket.pdb')
                # ligand_fn = os.path.join(pdb_path, f'{pdb_idx}_ligand.sdf')
                pocket_dict = PDBProtein(pocket_fn).to_dict_atom()
                ligand_dict = parse_sdf_file_mol(ligand_fn, heavy_only=self.heavy_only)
                data = ProteinLigandData.from_protein_ligand_dicts(
                    protein_dict=torchify_dict(pocket_dict),
                    ligand_dict=torchify_dict(ligand_dict),
                )
                data.protein_filename = pocket_fn
                data.ligand_filename = ligand_fn
                data.y = torch.tensor(float(pka))
                data.kind = torch.tensor(kind)
                txn.put(
                    key=f'{i:05d}'.encode(),
                    value=pickle.dumps(data)
                )
                # except:
                #     num_skipped += 1
                #     print('Skipping (%d) %s' % (num_skipped, ligand_fn, ))
                #     continue
        print('num_skipped: ', num_skipped)
    
    def __len__(self):
        if self.db is None:
            self._connect_db()
        return len(self.keys)

    def __getitem__(self, idx):
        if self.db is None:
            self._connect_db()
        key = self.keys[idx]
        data = pickle.loads(self.db.begin().get(key))
        data.id = idx
        assert data.protein_pos.size(0) > 0
        if self.transform is not None:
            data = self.transform(data)
        # add features extracted by molopt
        if self.emb_path is not None:
            emb = self.emb[idx]
            data.nll = torch.cat([emb['kl_pos'][1:], emb['kl_v'][1:]]).view(1, -1)
            data.nll_all = torch.cat([emb['kl_pos'], emb['kl_v']]).view(1, -1)
            data.pred_ligand_v = torch.softmax(emb['pred_ligand_v'], dim=-1)
            data.final_h = emb['final_h']
            # data.final_ligand_h = emb['final_ligand_h']
            data.pred_v_entropy = torch.from_numpy(
                stats.entropy(torch.softmax(emb['pred_ligand_v'], dim=-1).numpy(), axis=-1)).view(-1, 1)

        return data
        

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    args = parser.parse_args()

    PDBBindDataset(args.path)
