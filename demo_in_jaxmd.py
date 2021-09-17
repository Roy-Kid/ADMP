import jax
import jax.numpy as jnp
import numpy as np
from jax.api import grad, jit, vmap


from python.parser import assemble_covalent, init_residues, read_pdb, read_xml
from python.pme import (calc_ePermCoef, generate_construct_localframes,
                        rot_global2local, rot_local2global)
from python.utils import convert_cart2harm

# jax.profiler.start_trace("/tmp/tensorboard")
pdb = 'tests/samples/waterdimer_aligned.pdb'
# pdb = 'tests/samples/waterbox_31ang.pdb'
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

atomTemplate, residueTemplate = read_xml(xml)

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

covalent_map = assemble_covalent(residueDicts, natoms)    


kappa = 0.328532611


R = jnp.asarray(positions)

from jax_md import space
from jax_md import partition

rc = 4
box_size = 32
cell_size = rc
displacement_fn, shift_fn = space.periodic(box_size)
neighbor_list_fn = partition.neighbor_list(displacement_fn, box_size, cell_size, 0.2*rc)
neighbor = neighbor_list_fn(R)

calc_ePermCoef = jit(calc_ePermCoef)
rot_local2global = jit(rot_local2global, static_argnums=2)
rot_global2local = jit(rot_global2local, static_argnums=2)


def pme_real(dr, qiQI, qiQJ, kappa, mscales):


    cc, cd, dd_m0, dd_m1, cq, dq_m0, dq_m1, qq_m0, qq_m1, qq_m2 = calc_ePermCoef(mscales, kappa, dr)
    

    Vij0 = cc*qiQJ[:, 0]
    Vji0 = cc*qiQI[:, 0]

    # C-D 
    
    Vij0 = Vij0 - cd*qiQJ[:, 1]
    Vji1 = -cd*qiQI[:, 0]
    Vij1 = cd*qiQJ[:, 0]
    Vji0 = Vji0 + cd*qiQI[:, 1]

    # D-D m0 
    
    Vij1 += dd_m0 * qiQJ[:, 1]
    Vji1 += dd_m0 * qiQI[:, 1]
    
    # D-D m1 
    
    Vij2 = dd_m1*qiQJ[:, 2]
    Vji2 = dd_m1*qiQI[:, 2]
    Vij3 = dd_m1*qiQJ[:, 3]
    Vji3 = dd_m1*qiQI[:, 3]

    # C-Q
    
    Vij0 = Vij0 + cq*qiQJ[:, 4]
    Vji4 = cq*qiQI[:, 0]
    Vij4 = cq*qiQJ[:, 0]
    Vji0 = Vji0 + cq*qiQI[:, 4]
    
    # D-Q m0
    
    Vij1 += dq_m0*qiQJ[:, 4]
    Vji4 += dq_m0*qiQI[:, 1] 
    
    # Q-D m0
    
    Vij4 -= dq_m0*qiQJ[:, 1]
    Vji1 -= dq_m0*qiQI[:, 4]
    
    # D-Q m1
    
    Vij2 = Vij2 + dq_m1*qiQJ[:, 5]
    Vji5 = dq_m1*qiQI[:, 2]

    Vij3 += dq_m1*qiQJ[:, 6]
    Vji6 = dq_m1*qiQI[:, 3]
    
    Vij5 = -(dq_m1*qiQJ[:, 2])
    Vji2 += -(dq_m1*qiQI[:, 5])
    
    Vij6 = -(dq_m1*qiQJ[:, 3])
    Vji3 += -(dq_m1*qiQI[:, 6])
    
    # Q-Q m0
    
    Vij4 += qq_m0*qiQJ[:, 4]
    Vji4 += qq_m0*qiQI[:, 4] 
    
    # Q-Q m1
    
    Vij5 += qq_m1*qiQJ[:, 5]
    Vji5 += qq_m1*qiQI[:, 5]
    Vij6 += qq_m1*qiQJ[:, 6]
    Vji6 += qq_m1*qiQI[:, 6]
    
    # Q-Q m2
    
    Vij7  = qq_m2*qiQJ[:, 7]
    Vji7  = qq_m2*qiQI[:, 7]
    Vij8  = qq_m2*qiQJ[:, 8]
    Vji8  = qq_m2*qiQI[:, 8]
    
    Vij = jnp.vstack((Vij0, Vij1, Vij2, Vij3, Vij4, Vij5, Vij6, Vij7, Vij8))
    Vji = jnp.vstack((Vji0, Vji1, Vji2, Vji3, Vji4, Vji5, Vji6, Vji7, Vji8))

    
    return 0.5*(jnp.sum(qiQI*Vij.T)+jnp.sum(qiQJ*Vji.T))

jit_real = jit(pme_real)
grad_real = jit(grad(pme_real))

d = vmap(vmap(displacement_fn, (None, 0)))
mask = neighbor.idx != R.shape[0]
R_neigh = R[neighbor.idx]
dR = d(R, R_neigh)
# merged_kwargs = merge_dicts(kwargs, dynamic_kwargs)
# merged_kwargs = _neighborhood_kwargs_to_params(neighbor.idx,
#                                                 species,
#                                                 merged_kwargs,
#                                                 param_combinators)
# out = fn(dR, **merged_kwargs)
local_frames = generate_construct_localframes(axis_type, axis_indices)(positions, box)
Qglobal = rot_local2global(Qlocal, local_frames, 2)
mScales = jnp.concatenate([mScales, jnp.array([1])])
# pScales = jnp.concatenate([dynamic_kwargs['pScales'], jnp.array([1])])
# dScales = jnp.concatenate([dynamic_kwargs['dScales'], jnp.array([1])])


def qi(r1, r2):
    
    dr = displacement_fn(r1, r2)
    vectorZ = dr/jnp.linalg.norm(dr)
    # vectorX = jnp.array(vectorZ)
    # if r1[1] != r2[1] or r1[2] != r2[2]:
    #     vectorX = vectorX + jnp.array([1., 0., 0.])
    # else:
    #     vectorX = vectorX + jnp.array([0., 1., 0.])
    vectorX = jnp.where(jnp.logical_or(r1[1] != r2[1], r1[2]!=r2[2]), vectorZ+jnp.array([1., 0., 0.]), vectorZ + jnp.array([0., 1., 0.]))
    # vectorX = jax.lax.cond(r1[1] != r2[1] or r1[2] != r2[2],
    #                        lambda z: z+jnp.array([1., 0., 0.]),
    #                        lambda z: z+jnp.array([0., 1., 0.]),
    #                        operand=vectorZ)
    dot = jnp.dot(vectorZ, vectorX)
    vectorX -= vectorZ * dot
    vectorX = vectorX / jnp.linalg.norm(vectorX)
    vectorY = jnp.cross(vectorZ, vectorX)
    return jnp.array([vectorX, vectorY, vectorZ])

vvqi = vmap(vmap(qi, (None, 0)), (0, 0))
# vvqi = vmap(vmap(qi, (None, 0)), (0, None))
Ri = vvqi(R, R_neigh)
Ri = Ri[mask]
pairs = np.empty((Ri.shape[0], 2), dtype=int)
i = 0
for j, neighmk in enumerate(zip(neighbor.idx, mask)):
    neigh, mk = neighmk
    nei = neigh[mk]
    nnei = len(neigh[mk])
    pairs[i:i+nnei, 0] = j
    pairs[i:i+nnei, 1] = nei
    i += nnei

nbonds = covalent_map[pairs[:, 0], pairs[:, 1]]
mscales = mScales[nbonds - 1]
Q_extendi = Qglobal[pairs[:, 0]]
Q_extendj = Qglobal[pairs[:, 1]]
qiQI = rot_global2local(Q_extendi, Ri, 2)
qiQJ = rot_global2local(Q_extendj, Ri, 2)
distance = np.linalg.norm(dR[mask], axis=1)
e = pme_real(distance, qiQJ, qiQI, kappa = 0.328532611, mscales=mscales)
print(e)

# npair = len(pairs)
# dR = np.array(dR)
# R = np.array(R)
# for i in range(npair):
#     c, n = pairs[i]
#     dr = dR[i]
#     vectorZ = dr/np.linalg.norm(dr)
#     vectorX = np.array(vectorZ)
#     if R[c][1] != R[n][1] or R[c][2] != R[n][2]:
#         vectorX = vectorX + np.array([1., 0., 0.])
#     else:
#         vectorX = vectorX + np.array([0., 1., 0.])
#     dot = np.dot(vectorZ, vectorX)
#     vectorX -= vectorZ * dot
#     vectorX = vectorX / np.linalg.norm(vectorX)
#     vectorY = np.cross(vectorZ, vectorX)
#     Ri.append(np.array([vectorX, vectorY, vectorZ]))    
    


# Ri = jnp.asarray(Ri)

# Q_extendi = Qglobal[pairs[:, 0]]
# Q_extendj = Qglobal[pairs[:, 1]]

# qiQI = rot_global2local(Q_extendi, Ri, 2)
# qiQJ = rot_global2local(Q_extendj, Ri, 2)    
# distances = jnp.linalg.norm(dR, axis=1)

# # jit this
# # e = jit_real(distances, qiQJ, qiQI, kappa = 0.328532611, mscales=mscales)
# # f = grad_real(pme_real)(distances, qiQJ, qiQI, kappa = 0.328532611, mscales=mscales)
