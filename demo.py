from jax import vmap, jit, grad
from jax_md import partition
from jax_md import space
from jax.scipy.special import erf
from python.parser import assemble_covalent, init_residues, read_pdb, read_xml
from python.pme import generate_construct_localframes, rot_local2global, rot_global2local
from python.utils import convert_cart2harm
import jax
import jax.numpy as jnp
import numpy as np

def _calc_ePermCoef(mscales, kappa, dr):

    '''
    This function calculates the ePermCoefs at once

    Inputs:
        mscales:
            array: same as pme_realspace()
        kappa:
            float: same as pme_realspace()
        dr: 
            float: distance between one pair of particles
    Output:
        cc, cd, dd_m0, dd_m1, cq, dq_m0, dq_m1, qq_m0, qq_m1, qq_m2:
            n * 1 array: ePermCoefs
    '''

    # be aware of unit and dimension !!

    prefactor = dielectric
    
    rInvVec = jnp.array([prefactor/dr**i for i in range(0, 9)])
    
    alphaRVec = jnp.array([(kappa*dr)**i for i in range(0, 10)])
    
    X = 2 * jnp.exp( -alphaRVec[2] ) / jnp.sqrt(np.pi)
    tmp = jnp.array(alphaRVec[1])
    doubleFactorial = 1
    facCount = 1
    erfAlphaR = erf(alphaRVec[1])
    # bVec = jnp.empty((6, len(erfAlphaR)))
    # bVec = bVec.at[1].set(-erfAlphaR.reshape(-1))
    # for i in range(2, 6):
    #     bVec = bVec.at[i].set((bVec[i-1]+tmp*X/doubleFactorial))
    #     facCount += 2
    #     doubleFactorial *= facCount
    #     tmp *= 2 * alphaRVec[2]
        
    bVec = jnp.empty((6, len(erfAlphaR)))

    bVec = bVec.at[1].set(-erfAlphaR)
    for i in range(2, 6):
        bVec = bVec.at[i].set((bVec[i-1]+(tmp*X/doubleFactorial)))
        facCount += 2
        doubleFactorial *= facCount
        tmp *= 2 * alphaRVec[2]    
    
    # C-C: 1
    
    cc = rInvVec[1] * (mscales + bVec[2] - alphaRVec[1]*X) 
    
    # C-D: 1
    
    cd = rInvVec[2] * ( mscales + bVec[2] )
    
    ## D-D: 2
    
    dd_m0 = -2/3 * rInvVec[3] * (3*(mscales + bVec[3]) + alphaRVec[3]*X )
    
    dd_m1 = rInvVec[3] * (mscales + bVec[3] - (2/3)*alphaRVec[3]*X)
    
    ## C-Q: 1
    
    cq = (mscales + bVec[3])*rInvVec[3]
    
    ## D-Q: 2
    
    dq_m0 = rInvVec[4] * (3* (mscales + bVec[3])+ (4/3) * alphaRVec[5]*X)
    dq_m1 = -jnp.sqrt(3)*rInvVec[4]*(mscales+bVec[3])
    
    ## Q-Q
    
    qq_m0 = rInvVec[5] * (6* (mscales + bVec[4])+ (4/45)* (-3 + 10*alphaRVec[2]) * alphaRVec[5]*X)
    qq_m1 = -(4/15)*rInvVec[5]*(15*(mscales+bVec[4]) + alphaRVec[5]*X)
    qq_m2 = rInvVec[5]*(mscales + bVec[4] - (4/15)*alphaRVec[5]*X)

    return cc, cd, dd_m0, dd_m1, cq, dq_m0, dq_m1, qq_m0, qq_m1, qq_m2

def _pme_real(dr, qiQI, qiQJ, kappa, mscales):

    
    cc, cd, dd_m0, dd_m1, cq, dq_m0, dq_m1, qq_m0, qq_m1, qq_m2 = _calc_ePermCoef(mscales, kappa, dr)
    

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

jitted_rot_local2global = jit(rot_local2global, static_argnums=2)
jitted_rot_global2local = jit(rot_global2local, static_argnums=2)
_jitted_pme_real = jit(_pme_real)

def pme_real(positions, Qlocal, box, kappa, mScales, axis_type, axis_indices):

# --- build neighborlist ---
    rc = 4
    box_size = 32
    cell_size = rc
    displacement_fn, shift_fn = space.periodic(box_size)
    neighbor_list_fn = partition.neighbor_list(displacement_fn, box_size, cell_size, 0.2*rc)
    neighbor = neighbor_list_fn(positions)
# --- end build neighborlist ---

# --- convert Qlocal to Qglobal ---
    construct_localframes = generate_construct_localframes(axis_type, axis_indices)
    jitted_construct_localframes = jit(construct_localframes)
    local_frames = jitted_construct_localframes(positions, box)
    Qglobal = jitted_rot_local2global(Qlocal, local_frames, 2)
# --- end convert Qlocal to Qglobal ---

# --- build pair stack in numpy ---
    neighboridx = np.asarray(jax.device_get(neighbor.idx))
    mask = neighboridx != positions.shape[0]
    r = np.sum(mask, axis=1)
    pair1 = np.repeat(np.arange(neighboridx.shape[0]), r)
    pair2 = neighboridx[mask]
    pairs = np.stack((pair1, pair2), axis=1)
    pairs = pairs[pairs[:, 0]<pairs[:, 1]]
# --- end build pair stack in numpy ---

# --- build quasi internal in numpy ---

    r1 = positions[pairs[:, 0]]
    r2 = positions[pairs[:, 1]]

    d = vmap(displacement_fn)
    dr = d(r1, r2)
    norm_dr = jnp.linalg.norm(dr, axis=1)
    vectorZ = dr/norm_dr.reshape((-1, 1))

    vectorX = jnp.where(jnp.logical_or(r1[:, 1] != r2[:, 1], r1[:, 2]!=r2[:, 2]).reshape((-1, 1)), vectorZ+jnp.array([1., 0., 0.]), vectorZ + jnp.array([0., 1., 0.]))
    dot = vectorZ * vectorX
    vectorX -= vectorZ * dot
    vectorX = vectorX / jnp.linalg.norm(vectorX, axis=1).reshape((-1, 1))
    vectorY = jnp.cross(vectorZ, vectorX)
    Ri = jnp.stack([vectorX, vectorY, vectorZ], axis=1)
# --- end build quasi internal in numpy ---

# --- build coresponding Q matrix ---
    Q_extendi = Qglobal[pairs[:, 0]]
    Q_extendj = Qglobal[pairs[:, 1]]

    qiQI = jitted_rot_global2local(Q_extendi, Ri, 2)
    qiQJ = jitted_rot_global2local(Q_extendj, Ri, 2)

    nbonds = covalent_map[pairs[:, 0], pairs[:, 1]]
    mscales = mScales[nbonds-1]
# --- end build coresponding Q matrix ---

# --- actual calculation and jit ---
    return _pme_real(norm_dr, qiQJ, qiQI, kappa, mscales)

    
if __name__ == '__main__':
# --- prepare data ---

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

    mScales = np.array([0.0, 0.0, 0.0, 1.0, 1.0])
    pScales = np.array([0.0, 0.0, 0.0, 1.0, 1.0])
    dScales = np.array([0.0, 0.0, 0.0, 1.0, 1.0])

    box = np.eye(3)*np.array([lx, ly, lz])

    rc = 4 # in Angstrom
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
    dielectric = 1389.35455846 # in e^2/A

    import scipy
    from scipy.stats import special_ortho_group

    scipy.random.seed(1000)
    R1 = special_ortho_group.rvs(3)
    R2 = special_ortho_group.rvs(3)

    positions[0:3] = positions[0:3].dot(R1)
    positions[3:6] = positions[3:6].dot(R2)
    positions[3:] += np.array([3.0, 0.0, 0.0])
    positions = jnp.asarray(positions)


    # --- end prepare data ---
    jax.profiler.start_trace("./tensorboard")
    
    e = pme_real(positions, Qlocal, box, kappa, mScales, axis_type, axis_indices)
    print(e)
    # f = grad(pme_real)(positions, Qlocal, box, kappa, mScales, axis_type, axis_indices)
    # print(e, f)
    jax.profiler.stop_trace()