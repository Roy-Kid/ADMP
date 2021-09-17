# from python.neiV2 import construct_nblist as cnbl2

from python.utils import convert_cart2harm
import numpy as np
import jax.numpy as jnp
from python.parser import assemble_covalent, init_residues, read_pdb, read_xml
from python.pme import pme_real, generate_construct_localframes, rot_local2global, rot_global2local

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

local_frames = generate_construct_localframes(axis_type, axis_indices)(positions, box)
Qglobal = rot_local2global(Qlocal, local_frames, 2)

# e = pme_real(positions, box, rc, Qlocal, generate_construct_localframes(axis_type, axis_indices), kappa, covalent_map, mScales, pScales, dScales)
# print(e)

# print('auto diff: ')
# positions = jnp.asarray(positions)
# box = jnp.asarray(box)
# dreal = jax.grad(pme_real, argnums=0)
# f = dreal(positions, box, rc, Qlocal, generate_construct_localframes(axis_type, axis_indices), kappa, covalent_map, mScales, pScales, dScales)
# print(f)

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

from jax_md import space
from jax_md import partition
from jax import vmap
R = jnp.asarray(positions)
rc = 4
box_size = 32
cell_size = rc
displacement_fn, shift_fn = space.periodic(box_size)
neighbor_list_fn = partition.neighbor_list(displacement_fn, box_size, cell_size, 0.2*rc)
neighbor = neighbor_list_fn(R)

d = vmap(vmap(displacement_fn, (None, 0)))
mask = neighbor.idx != R.shape[0]
R_neigh = R[neighbor.idx]
dR = d(R, R_neigh)

def qi(r1, r2):
    
    dr = displacement_fn(r1, r2)
    vectorZ = dr/jnp.linalg.norm(dr)
    vectorX = jnp.where(jnp.logical_or(r1[1] != r2[1], r1[2]!=r2[2]), vectorZ+jnp.array([1., 0., 0.]), vectorZ + jnp.array([0., 1., 0.]))
    dot = jnp.dot(vectorZ, vectorX)
    vectorX -= vectorZ * dot
    vectorX = vectorX / jnp.linalg.norm(vectorX)
    vectorY = jnp.cross(vectorZ, vectorX)
    return jnp.array([vectorX, vectorY, vectorZ])

vvqi = vmap(vmap(qi, (None, 0)), (0, 0))
Ri = vvqi(R, R_neigh)
Ri = Ri[mask]

# so far is right.

pairs = np.empty((Ri.shape[0], 2), dtype=int)

i = 0
for j, neighmk in enumerate(zip(neighbor.idx, mask)):
    neigh, mk = neighmk
    nei = neigh[mk]
    nnei = len(neigh[mk])
    pairs[i:i+nnei, 0] = j
    pairs[i:i+nnei, 1] = nei
    i += nnei

pairs = pairs[pairs[:, 0]<pairs[:, 1]]
nbonds = covalent_map[pairs[:, 0], pairs[:, 1]]
mscales = mScales[nbonds - 1]
Q_extendi = Qglobal[pairs[:, 0]]
Q_extendj = Qglobal[pairs[:, 1]]
qiQI = rot_global2local(Q_extendi, Ri, 2)
qiQJ = rot_global2local(Q_extendj, Ri, 2)
distance = np.linalg.norm(dR[mask], axis=1)
