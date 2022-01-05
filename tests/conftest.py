
from admp.parser import *
import jax.numpy as jnp
import pytest
from admp.multipole import convert_cart2harm

def load(pdb_file, xml_file):
    
    pdbinfo = read_pdb(pdb_file)
    serials = pdbinfo['serials']
    n_atoms = len(serials)
    names = pdbinfo['names']
    resNames = pdbinfo['resNames']
    resSeqs = pdbinfo['resSeqs']
    positions = pdbinfo['positions']
    box = pdbinfo['box'] # a, b, c, α, β, γ
    charges = pdbinfo['charges']
    positions = jnp.asarray(positions)
    lx, ly, lz, _, _, _ = box
    box = jnp.eye(3)*jnp.array([lx, ly, lz])

    atomTemplate, residueTemplate = read_xml(xml_file)
    atomDicts, residueDicts = init_residues(serials, names, resNames, resSeqs, positions, charges, atomTemplate, residueTemplate)

    Q = np.vstack(
        [(atom.c0, atom.dX*10, atom.dY*10, atom.dZ*10, atom.qXX*300, atom.qYY*300, atom.qZZ*300, atom.qXY*300, atom.qXZ*300, atom.qYZ*300) for atom in atomDicts.values()]
    )
    Q = jnp.array(Q)

    axis_type = np.array(
        [atom.axisType for atom in atomDicts.values()]
    )
    axis_indices = np.vstack(
        [atom.axis_indices for atom in atomDicts.values()]
    )
    covalent_map = assemble_covalent(residueDicts, n_atoms)
    Q_local = convert_cart2harm(Q, 2)

    return serials, names, resNames, resSeqs, positions, box, charges, Q_local, axis_type, axis_indices, covalent_map

@pytest.fixture(scope='session', params=[1, 2], ids=lambda x: f'water{x}')
def water(request):
    print(f'load {request.param}')
    yield f'water{request.param}', load(f'water{request.param}.pdb', 'mpidwater.xml')