from admp.api import Hamiltonian
from admp.parser import *
import jax.numpy as jnp
import pytest
from admp.multipole import convert_cart2harm
from time import time
import openmm
import openmm.app as app
import openmm.unit as unit
from jax_md import space, partition

def load(pdb_file, xml_file):

    start = time()

    pdbinfo = read_pdb(pdb_file)
    serials = pdbinfo["serials"]
    n_atoms = len(serials)
    names = pdbinfo["names"]
    resNames = pdbinfo["resNames"]
    resSeqs = pdbinfo["resSeqs"]
    positions = pdbinfo["positions"]
    box = pdbinfo["box"]  # a, b, c, α, β, γ
    charges = pdbinfo["charges"]
    positions = jnp.asarray(positions)
    lx, ly, lz, _, _, _ = box
    box = jnp.eye(3) * jnp.array([lx, ly, lz])

    atomTemplate, residueTemplate = read_xml(xml_file)
    atomDicts, residueDicts = init_residues(
        serials,
        names,
        resNames,
        resSeqs,
        positions,
        charges,
        atomTemplate,
        residueTemplate,
    )

    Q = np.vstack(
        [
            (
                atom.c0,
                atom.dX * 10,
                atom.dY * 10,
                atom.dZ * 10,
                atom.qXX * 300,
                atom.qYY * 300,
                atom.qZZ * 300,
                atom.qXY * 300,
                atom.qXZ * 300,
                atom.qYZ * 300,
            )
            for atom in atomDicts.values()
        ]
    )
    Q = jnp.array(Q)

    axis_type = np.array([atom.axisType for atom in atomDicts.values()])
    axis_indices = np.vstack([atom.axis_indices for atom in atomDicts.values()])
    covalent_map = assemble_covalent(residueDicts, n_atoms)
    Q_local = convert_cart2harm(Q, 2)

    return (
        serials,
        names,
        resNames,
        resSeqs,
        positions,
        box,
        charges,
        Q,
        Q_local,
        axis_type,
        axis_indices,
        covalent_map,
    )


@pytest.fixture(scope="session", params=[1024], ids=lambda x: f"water{x}")
def water(request):
    yield f"{request.param}", load(f"water{request.param}.pdb", "mpidwater.xml")

# def openmmLoad(pdb_file, xml_file, residue_file):
    
#     H = Hamiltonian(xml_file)
    
#     generators = H.getGenerators()
#     app.Topology.loadBondDefinitions(residue_file)
#     pdb = app.PDBFile(pdb_file)
#     rc = 4.0
#     potentials = H.createPotential(pdb.topology, nonbondedCutoff=rc*unit.angstrom)
#     pot_disp = potentials[0]
#     pot_pme = potentials[1]

#     positions = jnp.array(pdb.positions._value) * 10
#     a, b, c = pdb.topology.getPeriodicBoxVectors()
#     box = jnp.array([a._value, b._value, c._value]) * 10        
#     mScales = generators[0].params['mScales']
#     displacement_fn, shift_fn = space.periodic_general(
#         box, fractional_coordinates=False
#     )
#     neighbor_list_fn = partition.neighbor_list(
#         displacement_fn, box, rc, 0, format=partition.OrderedSparse
#     )
#     nbr = neighbor_list_fn.allocate(positions)
#     pairs = nbr.idx.T
#     # positions, box, pairs, Q_local, mScales
#     E_, F_ = pot_pme(positions, box, pairs, generators[1].params)

#     # E, F = pme_force.get_forces(positions, box, pairs, generators[1].params['Q_local'], generators[1].params['mScales'])
#     return H, pdb

# @pytest.fixture(scope='session')
# def waterFromOpenMM():
#     return openmmLoad('water1024.pdb', 'mpidwater.xml', 'residue.xml')