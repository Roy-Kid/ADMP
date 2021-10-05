
import jax
import jax.numpy as jnp
import numpy as np
from python.neighborList import construct_nblist
from python.utils import rot_global2local, rot_local2global
import sys
sys.path.append('/home/roy/work/ADMP')
print(sys.path)
dielectric = 1389.35455846  # in e^2/A

mScales = np.array([0.0, 0.0, 0.0, 1.0, 1.0])
pScales = np.array([0.0, 0.0, 0.0, 1.0, 1.0])
dScales = np.array([0.0, 0.0, 0.0, 1.0, 1.0])
rc = 8  # in Angstrom
ethresh = 1e-4
kappa = 0.328532611
lmax = 2

local_frames = np.load('/home/roy/work/ADMP/python/local_frames.npy')
positions = np.load('/home/roy/work/ADMP/python/positions.npy')
box = jnp.asarray(np.load('/home/roy/work/ADMP/python/boxs.npy'))
Qlocal = np.load('/home/roy/work/ADMP/python/Qlocal.npy')
Qglobal = jnp.array(rot_local2global(Qlocal, local_frames, 2))
covalent_map = np.array(
    np.load('/home/roy/work/ADMP/python/covalent_map.npy', allow_pickle=True))
# neighlist = np.load('neighList.npy', allow_pickle=True)
neighlist = construct_nblist(positions, box, rc)


# the neighbor number of each center
n_nbs = neighlist.n_nbs

# max number in n_nbs
max_n_nbs = np.max(n_nbs)

# actual neighborlist
nbs = neighlist.nbs[:, 0:max_n_nbs]

# dr (vector) of pairs
dr_vecs_o = neighlist.dr_vecs[:, 0:max_n_nbs]
# dr^2 of pairs
distance2_o = neighlist.distances2[:, 0:max_n_nbs]
dr_o = distance2_o**0.5
box_inv = jnp.linalg.inv(box)
n_atoms = len(positions)
t = np.sum(n_nbs)
tot_pair = np.empty((t, 2), dtype=int)
dr = jnp.zeros((t, 3))
distance = jnp.empty((t, ))

for i_atom in range(n_atoms):

    start = np.sum(n_nbs[0:i_atom])
    end = np.sum(n_nbs[0:i_atom]) + n_nbs[i_atom]

    tot_pair[start: end, 0] = i_atom
    tot_pair[start: end, 1] = nbs[i_atom, 0: n_nbs[i_atom]]

    dr = dr.at[start: end, :].set(dr_vecs_o[i_atom, : n_nbs[i_atom]])
    distance = distance.at[start: end].set(dr_o[i_atom, : n_nbs[i_atom]])

filter = tot_pair[:, 0] < tot_pair[:, 1]
pairs = tot_pair[filter]
dr = dr[filter]
distance = distance[filter]

npair = len(pairs)

nbond = covalent_map[pairs[:, 0]][np.arange(len(pairs)), pairs[:, 1]]
mscales = mScales[nbond-1]

iPos = positions[pairs[:, 0]]
jPos = positions[pairs[:, 1]]

vectorZs = dr/distance.reshape((-1, 1))

vectorXs = jnp.array(vectorZs)

t1 = jnp.logical_or(iPos[:, 1] != jPos[:, 1], iPos[:, 2] != jPos[:, 2])
t2 = jnp.logical_not(t1)
vectorXs = vectorXs.at[t1, 0].add(1)
vectorXs = vectorXs.at[t2, 1].add(1)
dot = np.sum(vectorXs*vectorZs, axis=1)
vectorXs -= vectorZs*dot.reshape((-1, 1))
vectorXs = vectorXs / jnp.linalg.norm(vectorXs, axis=1).reshape((-1, 1))
vectorYs = jnp.cross(vectorZs, vectorXs)
Ri = jnp.stack((vectorXs, vectorYs, vectorZs), axis=1)

Q_extendi = Qglobal[pairs[:, 0]]
Q_extendj = Qglobal[pairs[:, 1]]

qiQI = jnp.array(rot_global2local(Q_extendi, Ri, 2))
qiQJ = jnp.array(rot_global2local(Q_extendj, Ri, 2))

prefactor = dielectric

rInvVecs = jnp.array([prefactor/distance**i for i in range(0, 9)])

alphaRVec = jnp.array([(kappa*distance)**i for i in range(0, 10)])
assert (alphaRVec[1] == kappa * distance).all()

X = 2 * jnp.exp(-alphaRVec[2]) / np.sqrt(np.pi)
tmp = alphaRVec[1]
doubleFactorial = 1
facCount = 1

erfAlphaR = jnp.vectorize(jax.scipy.special.erf)(alphaRVec[1])
bVec = jnp.empty((6, len(erfAlphaR)))
# warning
bVec = bVec.at[1].set(-erfAlphaR)
for i in range(2, 6):
    bVec = bVec.at[i].set(bVec[i-1]+tmp*X/doubleFactorial)
    facCount += 2
    doubleFactorial *= facCount
    tmp *= 2 * alphaRVec[2]

cc = rInvVecs[1] * (mscales + bVec[2] - alphaRVec[1]*X)
cd = rInvVecs[2] * (mscales + bVec[2])

# D-D: 2

dd_m0 = -2/3 * rInvVecs[3] * (3*(mscales + bVec[3]) + alphaRVec[3]*X)

dd_m1 = rInvVecs[3] * (mscales + bVec[3] - (2/3)*alphaRVec[3]*X)

# C-Q: 1

cq = (mscales + bVec[3])*rInvVecs[3]

# D-Q: 2

dq_m0 = rInvVecs[4] * (3 * (mscales + bVec[3]) + (4/3) * alphaRVec[5]*X)
dq_m1 = -np.sqrt(3)*rInvVecs[4]*(mscales+bVec[3])

# Q-Q

qq_m0 = rInvVecs[5] * (6 * (mscales + bVec[4]) + (4/45)
                       * (-3 + 10*alphaRVec[2]) * alphaRVec[5]*X)
qq_m1 = -(4/15)*rInvVecs[5]*(15*(mscales+bVec[4]) + alphaRVec[5]*X)
qq_m2 = rInvVecs[5]*(mscales + bVec[4] - (4/15)*alphaRVec[5]*X)

Vij = jnp.stack(

    (cc*qiQJ[:, 0]-cd*qiQJ[:, 1]+cq*qiQJ[:, 4],

     cd*qiQJ[:, 0]+dd_m0 * qiQJ[:, 1]+dq_m0*qiQJ[:, 4],

     dd_m1*qiQJ[:, 2]+dq_m1*qiQJ[:, 5],

     dd_m1*qiQJ[:, 3]+dq_m1*qiQJ[:, 6],

     cq*qiQJ[:, 0]-(dq_m0*qiQJ[:, 1])+qq_m0*qiQJ[:, 4],

     -(dq_m1*qiQJ[:, 2])+qq_m1*qiQJ[:, 5],

        -(dq_m1*qiQJ[:, 3])+qq_m1*qiQJ[:, 6],

        qq_m2*qiQJ[:, 7],

        qq_m2*qiQJ[:, 8]), axis=1

)


Vji = jnp.stack((

    cc*qiQI[:, 0]+cd*qiQI[:, 1]+cq*qiQI[:, 4],

    -cd*qiQI[:, 0]+dd_m0 * qiQI[:, 1]-dq_m0*qiQI[:, 4],

    dd_m1*qiQI[:, 2]-(dq_m1*qiQI[:, 5]),

    dd_m1*qiQI[:, 3]-(dq_m1*qiQI[:, 6]),

    cq*qiQI[:, 0]+dq_m0*qiQI[:, 1] + qq_m0*qiQI[:, 4],

    dq_m1*qiQI[:, 2]+qq_m1*qiQI[:, 5],

    dq_m1*qiQI[:, 3]+qq_m1*qiQI[:, 6],

    qq_m2*qiQI[:, 7],

    qq_m2*qiQI[:, 8]), axis=1

)
print(' --- q & v ---')
print(qiQI.sum(), qiQJ.sum(), Vji.sum(), Vij.sum())
print('----------------')
print(0.5*(np.sum(qiQI*Vij)+np.sum(qiQJ*Vji)))


def pme_real(positions, box, Qlocal, kappa, covalent_map, mScales, pScales, dScales):
    
    neighList = construct_nblist(positions, box, rc)
    
    print(type(neighList.dr_vecs))

    Qglobal = jnp.array(rot_local2global(Qlocal, local_frames, 2))

    # the neighbor number of each center
    n_nbs = neighList.n_nbs

    # max number in n_nbs
    max_n_nbs = np.max(n_nbs)

    # actual neighborlist
    nbs = neighlist.nbs[:, 0:max_n_nbs]

    # dr (vector) of pairs
    dr_vecs_o = neighlist.dr_vecs[:, 0:max_n_nbs]
    # dr^2 of pairs
    distance2_o = neighlist.distances2[:, 0:max_n_nbs]
    dr_o = distance2_o**0.5
    box_inv = jnp.linalg.inv(box)
    n_atoms = len(positions)
    t = np.sum(n_nbs)
    tot_pair = np.empty((t, 2), dtype=int)
    dr = jnp.zeros((t, 3))
    distance = jnp.empty((t, ))

    for i_atom in range(n_atoms):

        start = np.sum(n_nbs[0:i_atom])
        end = np.sum(n_nbs[0:i_atom]) + n_nbs[i_atom]

        tot_pair[start: end, 0] = i_atom
        tot_pair[start: end, 1] = nbs[i_atom, 0: n_nbs[i_atom]]

        dr = dr.at[start: end, :].set(dr_vecs_o[i_atom, : n_nbs[i_atom]])
        distance = distance.at[start: end].set(dr_o[i_atom, : n_nbs[i_atom]])

    filter = tot_pair[:, 0] < tot_pair[:, 1]
    pairs = tot_pair[filter]
    dr = dr[filter]
    distance = distance[filter]

    npair = len(pairs)

    nbond = covalent_map[pairs[:, 0]][np.arange(len(pairs)), pairs[:, 1]]
    mscales = mScales[nbond-1]

    iPos = positions[pairs[:, 0]]
    jPos = positions[pairs[:, 1]]

    vectorZs = dr/distance.reshape((-1, 1))

    vectorXs = jnp.array(vectorZs)

    t1 = jnp.logical_or(iPos[:, 1] != jPos[:, 1], iPos[:, 2] != jPos[:, 2])
    t2 = jnp.logical_not(t1)
    vectorXs = vectorXs.at[t1, 0].add(1)
    vectorXs = vectorXs.at[t2, 1].add(1)
    dot = jnp.sum(vectorXs*vectorZs, axis=1)
    vectorXs -= vectorZs*dot.reshape((-1, 1))
    vectorXs = vectorXs / jnp.linalg.norm(vectorXs, axis=1).reshape((-1, 1))
    vectorYs = jnp.cross(vectorZs, vectorXs)
    Ri = jnp.stack((vectorXs, vectorYs, vectorZs), axis=1)

    Q_extendi = Qglobal[pairs[:, 0]]
    Q_extendj = Qglobal[pairs[:, 1]]

    qiQI = jnp.array(rot_global2local(Q_extendi, Ri, 2))
    qiQJ = jnp.array(rot_global2local(Q_extendj, Ri, 2))

    prefactor = dielectric

    rInvVecs = jnp.array([prefactor/distance**i for i in range(0, 9)])

    alphaRVec = jnp.array([(kappa*distance)**i for i in range(0, 10)])
    assert (alphaRVec[1] == kappa * distance).all()

    X = 2 * jnp.exp(-alphaRVec[2]) / np.sqrt(np.pi)
    tmp = alphaRVec[1]
    doubleFactorial = 1
    facCount = 1

    erfAlphaR = jnp.vectorize(jax.scipy.special.erf)(alphaRVec[1])
    bVec = jnp.empty((6, len(erfAlphaR)))
    # warning
    bVec = bVec.at[1].set(-erfAlphaR)
    for i in range(2, 6):
        bVec = bVec.at[i].set(bVec[i-1]+tmp*X/doubleFactorial)
        facCount += 2
        doubleFactorial *= facCount
        tmp *= 2 * alphaRVec[2]

    cc = rInvVecs[1] * (mscales + bVec[2] - alphaRVec[1]*X)
    cd = rInvVecs[2] * (mscales + bVec[2])

    # D-D: 2

    dd_m0 = -2/3 * rInvVecs[3] * (3*(mscales + bVec[3]) + alphaRVec[3]*X)

    dd_m1 = rInvVecs[3] * (mscales + bVec[3] - (2/3)*alphaRVec[3]*X)

    # C-Q: 1

    cq = (mscales + bVec[3])*rInvVecs[3]

    # D-Q: 2

    dq_m0 = rInvVecs[4] * (3 * (mscales + bVec[3]) + (4/3) * alphaRVec[5]*X)
    dq_m1 = -jnp.sqrt(3)*rInvVecs[4]*(mscales+bVec[3])

    # Q-Q

    qq_m0 = rInvVecs[5] * (6 * (mscales + bVec[4]) + (4/45)
                           * (-3 + 10*alphaRVec[2]) * alphaRVec[5]*X)
    qq_m1 = -(4/15)*rInvVecs[5]*(15*(mscales+bVec[4]) + alphaRVec[5]*X)
    qq_m2 = rInvVecs[5]*(mscales + bVec[4] - (4/15)*alphaRVec[5]*X)

    Vij = jnp.stack(

        (cc*qiQJ[:, 0]-cd*qiQJ[:, 1]+cq*qiQJ[:, 4],

         cd*qiQJ[:, 0]+dd_m0 * qiQJ[:, 1]+dq_m0*qiQJ[:, 4],

         dd_m1*qiQJ[:, 2]+dq_m1*qiQJ[:, 5],

         dd_m1*qiQJ[:, 3]+dq_m1*qiQJ[:, 6],

         cq*qiQJ[:, 0]-(dq_m0*qiQJ[:, 1])+qq_m0*qiQJ[:, 4],

         -(dq_m1*qiQJ[:, 2])+qq_m1*qiQJ[:, 5],

            -(dq_m1*qiQJ[:, 3])+qq_m1*qiQJ[:, 6],

            qq_m2*qiQJ[:, 7],

            qq_m2*qiQJ[:, 8]), axis=1

    )

    Vji = jnp.stack((

        cc*qiQI[:, 0]+cd*qiQI[:, 1]+cq*qiQI[:, 4],

        -cd*qiQI[:, 0]+dd_m0 * qiQI[:, 1]-dq_m0*qiQI[:, 4],

        dd_m1*qiQI[:, 2]-(dq_m1*qiQI[:, 5]),

        dd_m1*qiQI[:, 3]-(dq_m1*qiQI[:, 6]),

        cq*qiQI[:, 0]+dq_m0*qiQI[:, 1] + qq_m0*qiQI[:, 4],

        dq_m1*qiQI[:, 2]+qq_m1*qiQI[:, 5],

        dq_m1*qiQI[:, 3]+qq_m1*qiQI[:, 6],

        qq_m2*qiQI[:, 7],

        qq_m2*qiQI[:, 8]), axis=1

    )
    return 0.5*(jnp.sum(qiQI*Vij)+jnp.sum(qiQJ*Vji))
