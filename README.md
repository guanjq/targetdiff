# 3D Equivariant Diffusion for Target-Aware Molecule Generation and Affinity Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/guanjq/targetdiff/blob/main/LICIENCE)


This repository is the official implementation of 3D Equivariant Diffusion for Target-Aware Molecule Generation and Affinity Prediction (ICLR 2023). [[PDF]](https://openreview.net/pdf?id=kJqXEPXMsE0) 

<p align="center">
  <img src="assets/overview.png" /> 
</p>

## Installation

### Dependency

The code has been tested in the following environment:


| Package           | Version   |
|-------------------|-----------|
| Python            | 3.8       |
| PyTorch           | 1.13.1    |
| CUDA              | 11.6      |
| PyTorch Geometric | 2.2.0     |
| RDKit             | 2022.03.2 |

### Install via Conda and Pip
```bash
conda create -n targetdiff python=3.8
conda activate targetdiff
conda install pytorch pytorch-cuda=11.6 -c pytorch -c nvidia
conda install pyg -c pyg
conda install rdkit openbabel tensorboard pyyaml easydict python-lmdb -c conda-forge

# For Vina Docking
pip install meeko==0.1.dev3 scipy pdb2pqr vina==1.2.2
python -m pip install git+https://github.com/Valdes-Tresanco-MS/AutoDockTools_py3
```
The code should work with PyTorch >= 1.9.0 and PyG >= 2.0. You can change the package version according to your need.

### (Alternatively) Install via Mamba
Install Mamba

```bash
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
bash Mambaforge-$(uname)-$(uname -m).sh  # accept all terms and install to the default location
rm Mambaforge-$(uname)-$(uname -m).sh  # (optionally) remove installer after using it
source ~/.bashrc  # alternatively, one can restart their shell session to achieve the same result
```

Create Mamba environment
```bash
mamba env create -f environment.yaml
conda activate targetdiff  # note: one still needs to use `conda` to (de)activate environments
```

## Data
The data used for training / evaluating the model are organized in the [data](https://drive.google.com/drive/folders/1j21cc7-97TedKh_El5E34yI8o5ckI7eK?usp=share_link) Google Drive folder.

To train the model from scratch, you need to download the preprocessed lmdb file and split file:
* `crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb`
* `crossdocked_pocket10_pose_split.pt`

To evaluate the model on the test set, you need to download the `test_set.zip`. It includes the original PDB files that will be used in Vina Docking.

If you want to process the dataset from scratch, you need to download CrossDocked2020 v1.1 from [here](https://bits.csb.pitt.edu/files/crossdock2020/), and run the scripts in `scripts/data_preparation`:
* `clean_crossdocked.py` will filter the original dataset and keep the ones with RMSD < 1A. (corresponds to `crossdocked_v1.1_rmsd1.0.tar.gz` in the drive)
* `extract_pockets.py` clip the original protein file to a 10A region around the binding molecule. (It will generate a `index.pkl` file, but you don't need that if you have downloaded .lmdb file.)
* `split_pl_dataset.py` split the training and test set.

## Training
### Training from scratch
```bash
python scripts/train_diffusion.py configs/training.yml
```
### Trained model checkpoint
https://drive.google.com/drive/folders/1-ftaIrTXjWFhw3-0Twkrs5m0yX6CNarz?usp=share_link

## Sampling
### Sampling for pockets in the testset
```bash
python scripts/sample_diffusion.py configs/sampling.yml --data_id {i} # Replace {i} with the index of the data. i should be between 0 and 99 for the testset.
```
You can also speed up sampling with multiple GPUs, e.g.:
```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/batch_sample_diffusion.sh configs/sampling.yml outputs 4 0 0
CUDA_VISIBLE_DEVICES=1 bash scripts/batch_sample_diffusion.sh configs/sampling.yml outputs 4 1 0
CUDA_VISIBLE_DEVICES=2 bash scripts/batch_sample_diffusion.sh configs/sampling.yml outputs 4 2 0
CUDA_VISIBLE_DEVICES=3 bash scripts/batch_sample_diffusion.sh configs/sampling.yml outputs 4 3 0
```

### Sampling from pdb file
To sample from a protein pocket (a 10A region around the reference ligand):
```bash
python scripts/sample_for_pocket.py configs/sampling.yml --pdb_path examples/1h36_A_rec_1h36_r88_lig_tt_docked_0_pocket10.pdb
```

## Evaluation
### Evaluation from sampling results
```bash
python scripts/evaluate_diffusion.py {OUTPUT_DIR} --docking_mode vina_score --protein_root data/test_set
```
The docking mode can be chosen from {qvina, vina_score, vina_dock, none}

Note: It will take some time to prepare pqdqt and pqr files when you run the evaluation code with vina_score/vina_dock docking mode for the first time.

### Evaluation from meta files
We provide the sampling results (also docked) of our model and CVAE, AR, Pocket2Mol baselines [here](https://drive.google.com/drive/folders/19imu-mlwrjnQhgbXpwsLgA17s1Rv70YS?usp=share_link).

| Metafile Name                   | Original Paper                                                                                      |
|---------------------------------|-----------------------------------------------------------------------------------------------------|
| crossdocked_test_vina_docked.pt | -                                                                                                   |
| cvae_vina_docked.pt             | [liGAN](https://arxiv.org/abs/2110.15200)                                                           |
| ar_vina_docked.pt               | [AR](https://proceedings.neurips.cc/paper/2021/hash/314450613369e0ee72d0da7f6fee773c-Abstract.html) |
| pocket2mol_vina_docked.pt       | [Pocket2Mol](https://proceedings.mlr.press/v162/peng22b.html)                                       |
| targetdiff_vina_docked.pt       | [TargetDiff](https://openreview.net/pdf?id=kJqXEPXMsE0)                                             |

You can directly evaluate from the meta file, e.g.:
```bash
python scripts/evaluate_from_meta.py sampling_results/targetdiff_vina_docked.pt --result_path eval_targetdiff
```

## Citation
```
@inproceedings{guan3d,
  title={3D Equivariant Diffusion for Target-Aware Molecule Generation and Affinity Prediction},
  author={Guan, Jiaqi and Qian, Wesley Wei and Peng, Xingang and Su, Yufeng and Peng, Jian and Ma, Jianzhu},
  booktitle={International Conference on Learning Representations},
  year={2023}
}
```
