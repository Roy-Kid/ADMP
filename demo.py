from python.utils import convert_cart2harm
import numpy as np
import jax.numpy as jnp
from python.parser import assemble_covalent, init_residues, read_pdb, read_xml
from python.pme import pme_real, generate_construct_localframes, rot_local2global

import jax
jax.profiler.start_trace("/tmp/tensorboard")

pdb = 'tests/samples/waterdimer_aligned.pdb'
xml = 'tests/samples/mpidwater.xml'

# return a dict with raw data from pdb
pdbinfo = read_pdb(pdb)

serials = pdbinfo['serials']
names = pdbinfo['names']
resNames = pdbinfo['resNames']
resSeqs = pdbinfo['resSeqs']
positions = pdbinfo['positions']
charges = pdbinfo['charges']
box = pdbinfo['box'] # a, b, c, α, β, γ
lx = box[0]
ly = box[1]
lz = box[2]

mScales = np.array([0.0, 0.0, 0.0, 1.0])
pScales = np.array([0.0, 0.0, 0.0, 1.0])
dScales = np.array([0.0, 0.0, 0.0, 1.0])

box = np.eye(3)*np.array([lx, ly, lz])

rc = 8 # in Angstrom
ethresh = 1e-4

natoms = len(serials)

# Here are the templates[dict] from xml. 
# atomTemplates are per-atom info,
# residueTemplates contain what atoms in the residue and how they connect. 
atomTemplate, residueTemplate = read_xml(xml)

# then we use the template to init atom and residue objects
# TODO: an atomManager and residueManager
atomDicts, residueDicts = init_residues(serials, names, resNames, resSeqs, positions, charges, atomTemplate, residueTemplate)

Q = np.vstack(
    [(atom.c0, atom.dX*10, atom.dY*10, atom.dZ*10, atom.qXX*300, atom.qYY*300, atom.qZZ*300, atom.qXY*300, atom.qXZ*300, atom.qYZ*300) for atom in atomDicts.values()]
)

Qlocal = convert_cart2harm(Q, 2)

axis_type = np.array(
    [atom.axisType for atom in atomDicts.values()]
)

axis_indices = np.vstack(
    [atom.axis_indices for atom in atomDicts.values()]
)
# covalent_map is simply as N*N matrix now
# remove sparse matrix
covalent_map = assemble_covalent(residueDicts, natoms)    


## TODO: setup kappa

kappa = 0.328532611

import scipy
from scipy.stats import special_ortho_group

scipy.random.seed(1000)
R1 = special_ortho_group.rvs(3)
R2 = special_ortho_group.rvs(3)

positions[0:3] = positions[0:3].dot(R1)
positions[3:6] = positions[3:6].dot(R2)
positions[3:] += np.array([3.0, 0.0, 0.0])

e = pme_real(positions, box, rc, Qlocal, generate_construct_localframes(axis_type, axis_indices), kappa, covalent_map, mScales, pScales, dScales)
print(e)

print('auto diff: ')
positions = jnp.asarray(positions)
box = jnp.asarray(box)
dreal = jax.grad(pme_real, argnums=0)
f = dreal(positions, box, rc, Qlocal, generate_construct_localframes(axis_type, axis_indices), kappa, covalent_map, mScales, pScales, dScales)
print(f)

# finite differences
# findiff = np.empty((6, 3))
# eps = 1e-4
# delta = np.zeros((6,3)) 
# for i in range(6): 
#   for j in range(3): 
#     delta[i][j] = eps 
#     findiff[i][j] = (pme_real(positions+delta/2, box, rc, Qlocal, generate_construct_localframes(axis_type, axis_indices), kappa, covalent_map, mScales, pScales, dScales) - pme_real(positions-delta/2, box, rc, Qlocal, generate_construct_localframes(axis_type, axis_indices), kappa, covalent_map, mScales, pScales, dScales))/eps
#     delta[i][j] = 0
# print('partial diff')
# print(findiff)

jax.profiler.stop_trace()
