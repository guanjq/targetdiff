import os
import py3Dmol
from rdkit import Chem


def visualize_complex(pdb_block, sdf_block, show_protein_surface=True, show_ligand=True, show_ligand_surface=True):
    view = py3Dmol.view()

    # Add protein to the canvas
    view.addModel(pdb_block, 'pdb')
    if show_protein_surface:
        view.addSurface(py3Dmol.VDW, {'opacity': 0.7, 'color': 'white'}, {'model': -1})
    else:
        view.setStyle({'model': -1}, {'cartoon': {'color': 'spectrum'}, 'line': {}})
    view.setStyle({'model': -1}, {"cartoon": {"style": "edged", 'opacity': 0}})

    # Add ligand to the canvas
    if show_ligand:
        view.addModel(sdf_block, 'sdf')
        view.setStyle({'model': -1}, {'stick': {}})
        # view.setStyle({'model': -1}, {'cartoon': {'color': 'spectrum'}, 'line': {}})
        if show_ligand_surface:
            view.addSurface(py3Dmol.VDW, {'opacity': 0.8}, {'model': -1})

    view.zoomTo()
    return view


def visualize_data(data, root, show_ligand=True, show_surface=True):
    protein_path = os.path.join(root, data.protein_filename)
    ligand_path = os.path.join(root, data.ligand_filename)
    with open(protein_path, 'r') as f:
        pdb_block = f.read()
    with open(ligand_path, 'r') as f:
        sdf_block = f.read()
    return visualize_complex(pdb_block, sdf_block, show_ligand=show_ligand, show_surface=show_surface)


def visualize_generated_mol(protein_filename, mol, root, show_surface=False, opacity=0.5):
    protein_path = os.path.join(root, protein_filename)
    with open(protein_path, 'r') as f:
        pdb_block = f.read()

    view = py3Dmol.view()

    # Add protein to the canvas
    view.addModel(pdb_block, 'pdb')
    view.setStyle({'model': -1}, {'cartoon': {'color': 'spectrum'}, 'line': {}})

    mblock = Chem.MolToMolBlock(mol)
    view.addModel(mblock, 'mol')
    view.setStyle({'model': -1}, {'stick': {}, 'sphere': {'radius': 0.35}})
    if show_surface:
        view.addSurface(py3Dmol.SAS, {'opacity': opacity}, {'model': -1})

    view.zoomTo()
    return view


def MolTo3DView(mol, size=(300, 300), style="stick", surface=False, opacity=0.5):
    """Draw molecule in 3D

    Args:
    ----
        mol: rdMol, molecule to show
        size: tuple(int, int), canvas size
        style: str, type of drawing molecule
               style can be 'line', 'stick', 'sphere', 'carton'
        surface, bool, display SAS
        opacity, float, opacity of surface, range 0.0-1.0
    Return:
    ----
        viewer: py3Dmol.view, a class for constructing embedded 3Dmol.js views in ipython notebooks.
    """
    assert style in ('line', 'stick', 'sphere', 'carton')

    viewer = py3Dmol.view(width=size[0], height=size[1])
    if isinstance(mol, list):
        for i, m in enumerate(mol):
            mblock = Chem.MolToMolBlock(m)
            viewer.addModel(mblock, 'mol' + str(i))
    elif len(mol.GetConformers()) > 1:
        for i in range(len(mol.GetConformers())):
            mblock = Chem.MolToMolBlock(mol, confId=i)
            viewer.addModel(mblock, 'mol' + str(i))
    else:
        mblock = Chem.MolToMolBlock(mol)
        viewer.addModel(mblock, 'mol')
    viewer.setStyle({style: {}})
    if surface:
        viewer.addSurface(py3Dmol.SAS, {'opacity': opacity})
    viewer.zoomTo()
    return viewer
