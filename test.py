#!/usr/bin/env python
import sys
import numpy as np
import jax.numpy as jnp
from jax.scipy.special import erf
from admp.settings import *

# for debugging use only
from jax_md import partition, space
from admp.parser import *
from admp.multipole import *
from admp.spatial import *
from admp.pme import energy_pme
from jax import grad, value_and_grad


pdb = str(sys.argv[1])
xml = 'mpidwater.xml'
pdbinfo = read_pdb(pdb)
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

mScales = jnp.array([0.0, 0.0, 0.0, 1.0, 1.0])
pScales = jnp.array([0.0, 0.0, 0.0, 1.0, 1.0])
dScales = jnp.array([0.0, 0.0, 0.0, 1.0, 1.0])

rc = 4  # in Angstrom
ethresh = 1e-4

n_atoms = len(serials)

atomTemplate, residueTemplate = read_xml(xml)
atomDicts, residueDicts = init_residues(serials, names, resNames, resSeqs, positions, charges, atomTemplate, residueTemplate)

Q = np.vstack(
    [(atom.c0, atom.dX*10, atom.dY*10, atom.dZ*10, atom.qXX*300, atom.qYY*300, atom.qZZ*300, atom.qXY*300, atom.qXZ*300, atom.qYZ*300) for atom in atomDicts.values()]
)
Q = jnp.array(Q)
Q_local = convert_cart2harm(Q, 2)
axis_type = np.array(
    [atom.axisType for atom in atomDicts.values()]
)
axis_indices = np.vstack(
    [atom.axis_indices for atom in atomDicts.values()]
)
covalent_map = assemble_covalent(residueDicts, n_atoms)


displacement_fn, shift_fn = space.periodic_general(box, fractional_coordinates=False)
neighbor_list_fn = partition.neighbor_list(displacement_fn, box, rc, 0, format=partition.OrderedSparse)
nbr = neighbor_list_fn.allocate(positions)
pairs = nbr.idx.T
# pairs = pairs[pairs[:, 0] < pairs[:, 1]]

# invoke pme energy calculator
# energy_pme(positions, box, Q_local, axis_type, axis_indices, nbr.idx, 2)
lmax = 2
# kappa, K1, K2, K3 = setup_ewald_parameters(rc, ethresh, box)
# for debugging
kappa = 0.657065221219616
K1 = K2 = K3 = 97
construct_local_frames_fn = generate_construct_local_frames(axis_type, axis_indices)
energy_force_pme = value_and_grad(energy_pme)
e, f = energy_force_pme(positions, box, pairs, Q_local, mScales, pScales, dScales, covalent_map, construct_local_frames_fn, kappa, K1, K2, K3, lmax)
print('ok')
e, f = energy_force_pme(positions, box, pairs, Q_local, mScales, pScales, dScales, covalent_map, construct_local_frames_fn, kappa, K1, K2, K3, lmax)
print(e)
