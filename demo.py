from jax import vmap, jit, grad
from jax_md import partition
from jax_md import space
from jax.scipy.special import erf
from python.parser import assemble_covalent, init_residues, read_pdb, read_xml
import jax
import jax.numpy as jnp
import numpy as np
import time 
import sys
from jax.config import config
config.update("jax_enable_x64", True)
def convert_cart2harm(Theta, lmax):
    '''
    Convert the multipole moments in cartesian repr to spherical harmonic repr
    Essentially, implementing the equations in Appendix E in Anthony's book

    Inputs:
        Theta:
            n * N_cart: stores the cartesian multipoles of each site
        lmax:
            integer, the maximum l, currently only supports up to quadrupole

    Outputs:
        Q:
            n * (l+1)^2, stores the spherical multipoles
    '''
    if lmax > 2:
        sys.exit('l > 2 (beyond quadrupole) not supported')

    n_sites = Theta.shape[0]
    n_harm = (lmax + 1)**2
    Q = np.zeros((n_sites, n_harm))
    Q[:,0] = Theta[:,0]
    
    # dipole
    if lmax >= 1:
        dipoles_cart = Theta[:, 1:4].T
        C1 = np.zeros((3, 3))
        C1[0, 1] = 1
        C1[1, 2] = 1
        C1[2, 0] = 1
        Q[:, 1:4] = C1.T.dot(dipoles_cart).T
    # quadrupole
    if lmax >= 2:
        inv_rt3 = 1.0 / np.sqrt(3)
        C2_c2h = np.zeros((5, 6))
        C2_c2h[0, 2] = 1.0
        C2_c2h[1, 4] = 2 * inv_rt3
        C2_c2h[2, 5] = 2 * inv_rt3
        C2_c2h[3, 0] = inv_rt3
        C2_c2h[3, 1] = -inv_rt3
        C2_c2h[4, 3] = 2 * inv_rt3
        quad_cart = Theta[:, 4:10].T
        Q[:, 4:9] = C2_c2h.dot(quad_cart).T
    return Q

def rot_global2local(Q_gh, localframes, lmax = 2):
    '''
    This one rotate harmonic moments Q from global frame to local frame

    Input:
        Q_gh: 
            n * (l+1)^2, stores the global harmonic multipole moments of each site
        localframes: 
            n * 3 * 3, stores the Rotation matrix for each site, the R is defined as:
            [r1, r2, r3]^T, with r1, r2, r3 being the local frame axes
        lmax:
            integer, the maximum multipole order

    Output:
        Qrot:
            n * (l+1)^2, stores the rotated multipole moments
    '''
    if lmax > 2:
        raise NotImplementedError('l > 2 (beyond quadrupole) not supported')

    # monopole
    Q_lh_0 = Q_gh[:, 0]
    # for dipole
    if lmax >= 1:
        zxy = np.array([2,0,1])
        # the rotation matrix
        R1 = localframes[:, zxy][:,:,zxy]
        # rotate
        Q_lh_1 = np.sum(R1*Q_gh[:,np.newaxis,1:4], axis = 2)
    if lmax >= 2:
        rt3 = np.sqrt(3)
        xx = localframes[:, 0, 0]
        xy = localframes[:, 0, 1]
        xz = localframes[:, 0, 2]
        yx = localframes[:, 1, 0]
        yy = localframes[:, 1, 1]
        yz = localframes[:, 1, 2]
        zx = localframes[:, 2, 0]
        zy = localframes[:, 2, 1]
        zz = localframes[:, 2, 2]
        quadrupoles = Q_gh[:, 4:9]
        # construct the local->global transformation matrix
        # this is copied directly from the convert_mom_to_xml.py code
        C2_gl_00 = (3*zz**2-1)/2
        C2_gl_01 = rt3*zx*zz
        C2_gl_02 = rt3*zy*zz
        C2_gl_03 = (rt3*(-2*zy**2-zz**2+1))/2
        C2_gl_04 = rt3*zx*zy
        C2_gl_10 = rt3*xz*zz
        C2_gl_11 = 2*xx*zz-yy
        C2_gl_12 = yx+2*xy*zz
        C2_gl_13 = -2*xy*zy-xz*zz
        C2_gl_14 = xx*zy+zx*xy
        C2_gl_20 = rt3*yz*zz
        C2_gl_21 = 2*yx*zz+xy
        C2_gl_22 = -xx+2*yy*zz
        C2_gl_23 = -2*yy*zy-yz*zz
        C2_gl_24 = yx*zy+zx*yy
        C2_gl_30 = rt3*(-2*yz**2-zz**2+1)/2
        C2_gl_31 = -2*yx*yz-zx*zz
        C2_gl_32 = -2*yy*yz-zy*zz
        C2_gl_33 = (4*yy**2+2*zy**2+2*yz**2+zz**2-3)/2
        C2_gl_34 = -2*yx*yy-zx*zy
        C2_gl_40 = rt3*xz*yz
        C2_gl_41 = xx*yz+yx*xz
        C2_gl_42 = xy*yz+yy*xz
        C2_gl_43 = -2*xy*yy-xz*yz
        C2_gl_44 = xx*yy+yx*xy
        # rotate
        C2_gl = jnp.array(
            [
                [C2_gl_00, C2_gl_10, C2_gl_20, C2_gl_30, C2_gl_40],
                [C2_gl_01, C2_gl_11, C2_gl_21, C2_gl_31, C2_gl_41],
                [C2_gl_02, C2_gl_12, C2_gl_22, C2_gl_32, C2_gl_42],
                [C2_gl_03, C2_gl_13, C2_gl_23, C2_gl_33, C2_gl_43],
                [C2_gl_04, C2_gl_14, C2_gl_24, C2_gl_34, C2_gl_44]
            ]
        ).swapaxes(0,2)
        Q_lh_2 = jnp.einsum('ijk,ik->ij', C2_gl, quadrupoles)
    Q_lh = jnp.hstack([Q_lh_0[:,np.newaxis], Q_lh_1, Q_lh_2])

    return Q_lh

def rot_local2global(Q_lh, localframes, lmax = 2):
    return rot_global2local(Q_lh, jnp.swapaxes(localframes, 1, 2), lmax)

def generate_construct_localframes(axis_types, axis_indices):
    """
    Generates the local frame constructor, common to the same physical system
    inputs:
        axis_types:
            types of local frame transformation rules for each atom.
        axis_indices:
            z,x,y atoms of each local frame.
    outputs:
        construct_localframes:
            function type (positions, box) -> local_frames
    """
    ZThenX            = 0
    Bisector          = 1
    ZBisect           = 2
    ThreeFold         = 3
    Zonly             = 4
    NoAxisType        = 5
    LastAxisTypeIndex = 6
    
    z_atoms = jnp.array(axis_indices[:, 0])
    x_atoms = jnp.array(axis_indices[:, 1])
    y_atoms = jnp.array(axis_indices[:, 2])
    
    Zonly_filter = (axis_types == Zonly)
    not_Zonly_filter = jnp.logical_not(Zonly_filter)
    Bisector_filter = (axis_types == Bisector)
    ZBisect_filter = (axis_types == ZBisect)
    ThreeFold_filter = (axis_types == ThreeFold)
    
    def pbc_shift(drvecs, box, box_inv):
        '''
        Dealing with the pbc shifts of vectors

        Inputs:
            rvecs:
                N * 3, a list of real space vectors in Cartesian
            box:
                3 * 3, box matrix, with axes arranged in rows
            box_inv:
                3 * 3, inverse of box matrix

        Outputs:
            rvecs:
                N * 3, vectors that have been shifted, in Cartesian
        '''
        unshifted_dsvecs = drvecs.dot(box_inv)
        dsvecs = unshifted_dsvecs - jnp.floor(unshifted_dsvecs + 0.5)
        return dsvecs.dot(box)
        
    def normalize(matrix, axis=1, ord=2):
        '''
        Normalise a matrix along one dimension
        '''
        normalised = matrix / jnp.linalg.norm(matrix, axis=axis, keepdims=True, ord=ord)
        return normalised

    def c_l_f(positions, box):
        '''
        This function constructs the local frames for each site

        Inputs:
            positions:
                N * 3: the positions matrix
            box:
        Outputs:
            Q: 
                N*(lmax+1)^2, the multipole moments in global harmonics.
            local_frames:
                N*3*3, the local frames, axes arranged in rows
        '''

        positions = jnp.array(positions)
        n_sites = positions.shape[0]
        box_inv = jnp.linalg.inv(box)

        ### Process the x, y, z vectors according to local axis rules
        vec_z = pbc_shift(positions[z_atoms] - positions, box, box_inv)
        vec_z = normalize(vec_z)
        vec_x = jnp.zeros((n_sites, 3))
        vec_y = jnp.zeros((n_sites, 3))
        # Z-Only
        x_of_vec_z = jnp.round(jnp.abs(vec_z[:,0]))
        vec_x_Zonly = jnp.array([1.-x_of_vec_z, x_of_vec_z, jnp.zeros_like(x_of_vec_z)]).T
        vec_x = vec_x.at[Zonly_filter].set(vec_x_Zonly)
        # for those that are not Z-Only, get normalized vecX
        vec_x_not_Zonly = positions[x_atoms[not_Zonly_filter]] - positions[not_Zonly_filter]
        vec_x_not_Zonly = pbc_shift(vec_x_not_Zonly, box, box_inv)

        vec_x = vec_x.at[not_Zonly_filter].set(normalize(vec_x_not_Zonly, axis=1))
        # Bisector
        if np.sum(Bisector_filter) > 0:
            vec_z_Bisector = vec_z[Bisector_filter] + vec_x[Bisector_filter]
            vec_z = vec_z.at[Bisector_filter].set(normalize(vec_z_Bisector, axis=1))
        # z-bisector
        if np.sum(ZBisect_filter) > 0:
            vec_y_ZBisect = positions[y_atoms[ZBisect_filter]] - positions[ZBisect_filter]
            vec_y_ZBisect = pbc_shift(vec_y_ZBisect, box, box_inv)
            vec_y_ZBisect = normalize(vec_y_ZBisect, axis=1)
            vec_x_ZBisect = vec_x[ZBisect_filter] + vec_y_ZBisect
            vec_x = vec_x.at[ZBisect_filter].set(normalize(vec_x_ZBisect, axis=1))
        # ThreeFold
        if np.sum(ThreeFold_filter) > 0:
            vec_x_threeFold = vec_x[ThreeFold_filter]
            vec_z_threeFold = vec_z[ThreeFold_filter]
            
            vec_y_threeFold = positions[y_atoms[ThreeFold_filter]] - positions[ThreeFold_filter]
            vec_y_threeFold = pbc_shift(vec_y_threeFold, box, box_inv)
            vec_y_threeFold = normalize(vec_y_threeFold, axis=1)
            vec_z_threeFold += (vec_x_threeFold + vec_y_threeFold)
            vec_z_threeFold = normalize(vec_z_threeFold)
            
            vec_y = vec_y.at[ThreeFold_filter].set(vec_y_threeFold)
            vec_z = vec_z.at[ThreeFold_filter].set(vec_z_threeFold)
        
        
        # up to this point, z-axis should already be set up and normalized
        xz_projection = jnp.sum(vec_x*vec_z, axis = 1, keepdims=True)
        vec_x = normalize(vec_x - vec_z * xz_projection, axis=1)
        # up to this point, x-axis should be ready
        vec_y = jnp.cross(vec_z, vec_x)

        return jnp.stack((vec_x, vec_y, vec_z), axis=1)
    return c_l_f

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

t0 = time.time()

jitted_rot_local2global = jit(rot_local2global, static_argnums=2)
jitted_rot_global2local = jit(rot_global2local, static_argnums=2)
_jitted_pme_real = jit(_pme_real)

t1 = time.time()
print(f'jit cost: {t1-t0}')

def pme_real(positions, Qlocal, box, kappa, mScales):

# --- build neighborlist ---
    t = time.time()
    neighbor = neighbor_list_fn(positions)
    tt = time.time()
    print(f'summary: (sec)')
    print(f'neighbor list : {tt-t}')
# --- end build neighborlist ---

# --- convert Qlocal to Qglobal ---

    local_frames = jitted_construct_localframes(positions, box)
    Qglobal = jitted_rot_local2global(Qlocal, local_frames, 2)
    ttt = time.time()
    print(f'Qlocal to Qglobal: {ttt-tt}')
# --- end convert Qlocal to Qglobal ---

# --- build pair stack in numpy ---
    t0 = time.time()
    neighboridx = np.asarray(jax.device_get(neighbor.idx))
    mask = neighboridx != positions.shape[0]
    r = np.sum(mask, axis=1)
    pair1 = np.repeat(np.arange(neighboridx.shape[0]), r)
    pair2 = neighboridx[mask]
    pairs = np.stack((pair1, pair2), axis=1)
    pairs = pairs[pairs[:, 0]<pairs[:, 1]]
    t1 = time.time()
    print(f'pair stack cost: {t1-t0}')
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
    t2 = time.time()
    print(f'quasi internal cost: {t2 - t1}')
# --- end build quasi internal in numpy ---

# --- build coresponding Q matrix ---
    Q_extendi = Qglobal[pairs[:, 0]]
    Q_extendj = Qglobal[pairs[:, 1]]

    qiQI = jitted_rot_global2local(Q_extendi, Ri, 2)
    qiQJ = jitted_rot_global2local(Q_extendj, Ri, 2)

    nbonds = covalent_map[pairs[:, 0], pairs[:, 1]]
    mscales = mScales[nbonds-1]
    t3 = time.time()
    print(f'build qiQI/qiQJ matrix: {t3-t2}')
# --- end build coresponding Q matrix ---

# --- actual calculation and jit ---
    e = _jitted_pme_real(norm_dr, qiQJ, qiQI, kappa, mscales)
    t4 = time.time()
    print(f'actual calculation cost: {t4 -t3}')
    print(f'total time cost: {t4-t}')
    return e

    
if __name__ == '__main__':
# --- prepare data ---

    pdb = 'tests/samples/waterdimer_aligned.pdb'
    pdb = 'tests/samples/waterbox_31ang.pdb'
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
    print(f'n of atoms: {natoms}')

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

    construct_localframes = generate_construct_localframes(axis_type, axis_indices)
    jitted_construct_localframes = jit(construct_localframes)
    
    rc = 4
    box_size = 32
    cell_size = rc
    displacement_fn, shift_fn = space.periodic(box_size)
    neighbor_list_fn = partition.neighbor_list(displacement_fn, box_size, cell_size, 0.2*rc)
    displacement_fn = jit(displacement_fn)
    # --- end prepare data ---
    # jax.profiler.start_trace("./tensorboard")
    pme_force = grad(pme_real)
    for i in range(3):
        print(f'iter {i}')
        e = pme_real(positions, Qlocal, box, kappa, mScales)
        positions = positions + jnp.array([0.0001, 0.0001, 0.0001])
        f = pme_force(positions, Qlocal, box, kappa, mScales)
    # print(e, f)
    # jax.profiler.stop_trace()