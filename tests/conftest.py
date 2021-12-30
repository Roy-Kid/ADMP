import pytest
import jax.numpy as jnp
import numpy as np
from admp.parser import read_pdb, read_xml, init_residues, assemble_covalent

def load(pdbName, xmlName):
    pdbinfo = read_pdb(pdbName)
    serials = pdbinfo['serials']
    names = pdbinfo['names']
    resNames = pdbinfo['resNames']
    resSeqs = pdbinfo['resSeqs']
    positions = pdbinfo['positions']
    box = pdbinfo['box'] # a, b, c, α, β, γ
    charges = pdbinfo['charges']
    positions = jnp.asarray(positions)
    lx, ly, lz, _, _, _ = box
    box = jnp.eye(3)*jnp.array([lx, ly, lz])    
    n_atoms = len(serials)
    
    atomTemplate, residueTemplate = read_xml(xmlName)
    atomDicts, residueDicts = init_residues(serials, names, resNames, resSeqs, positions, charges, atomTemplate, residueTemplate)
    
    Q = jnp.vstack(
        [(atom.c0, atom.dX*10, atom.dY*10, atom.dZ*10, atom.qXX*300, atom.qYY*300, atom.qZZ*300, atom.qXY*300, atom.qXZ*300, atom.qYZ*300) for atom in atomDicts.values()]
    )
    # Q = jnp.array(Q)    
    axis_types = np.array(
        [atom.axisType for atom in atomDicts.values()]
    )
    axis_indices = np.vstack(
        [atom.axis_indices for atom in atomDicts.values()]
    )
    covalent_map = assemble_covalent(residueDicts, n_atoms)
    
    return positions, box, atomDicts, axis_types, axis_indices, covalent_map

# jichen: usage of param fixture can refer to
# https://stackoverflow.com/questions/42014484/pytest-using-fixtures-as-arguments-in-parametrize
@pytest.fixture(scope='package')
def water1():
    return load(f'water1.pdb', 'mpidwater.xml')
@pytest.fixture(scope='package')
def water2():
    return load(f'water2.pdb', 'mpidwater.xml')
@pytest.fixture(scope='package')
def water4():
    return load(f'water4.pdb', 'mpidwater.xml')
@pytest.fixture(scope='package')
def water8():
    return load(f'water8.pdb', 'mpidwater.xml')
@pytest.fixture(scope='package')
def water256():
    return load(f'water256.pdb', 'mpidwater.xml')
@pytest.fixture(scope='package')
def water512():
    return load(f'water512.pdb', 'mpidwater.xml')
@pytest.fixture(scope='package')
def water1024():
    return load(f'water1024.pdb', 'mpidwater.xml')
        