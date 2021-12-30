#!/usr/bin/env python
import sys
import numpy as np
import jax.numpy as jnp
from jax import grad, value_and_grad
from jax.scipy.special import erf
from admp.settings import *
from admp.multipole import *
from admp.spatial import *

DIELECTRIC = 1389.35455846
from admp.recip import *

# for debugging use only
from jax_md import partition, space
from admp.parser import *

# from jax.config import config
# config.update("jax_enable_x64", True)

# Functions that are related to electrostatic pme

class ADMPPmeForce:
    '''
    This is a convenient wrapper for multipolar PME calculations
    It wrapps all the environment parameters of multipolar PME calculation
    The so called "environment paramters" means parameters that do not need to be differentiable
    '''

    def __init__(self, box, axis_type, axis_indices, covalent_map, rc, ethresh, lmax):
        self.axis_type = axis_type
        self.axis_indices = axis_indices
        self.rc = rc
        self.ethresh = ethresh
        self.lmax = lmax
        kappa, K1, K2, K3 = setup_ewald_parameters(rc, ethresh, box)
        self.kappa = kappa
        self.K1 = K1
        self.K2 = K2
        self.K3 = K3
        self.pme_order = 6
        self.covalent_map = covalent_map

        # setup calculators
        self.refresh_calculators()
        return


    def generate_get_energy(self):
        def get_energy(positions, box, pairs, Q_local, mScales, pScales, dScales):
            return energy_pme(positions, box, pairs,
                             Q_local, mScales, pScales, dScales, self.covalent_map,
                             self.construct_local_frames, self.pme_recip,
                             self.kappa, self.K1, self.K2, self.K3, self.lmax)
        return get_energy


    def update_env(self, attr, val):
        '''
        Update the environment of the calculator
        '''
        setattr(self, attr, val)
        self.refresh_calculators()


    def refresh_calculators(self):
        '''
        refresh the energy and force calculators according to the current environment
        '''
        self.construct_local_frames = generate_construct_local_frames(self.axis_type, self.axis_indices)
        self.pme_recip = generate_pme_recip(Ck_1, self.kappa, False, self.pme_order, self.K1, self.K2, self.K3, self.lmax)
        # generate the force calculator
        self.get_energy = self.generate_get_energy()
        self.get_forces = value_and_grad(self.get_energy)
        return


def setup_ewald_parameters(rc, ethresh, box):
    '''
    Given the cutoff distance, and the required precision, determine the parameters used in
    Ewald sum, including: kappa, K1, K2, and K3.
    The algorithm is exactly the same as OpenMM, see: 
    http://docs.openmm.org/latest/userguide/theory.html

    Input:
        rc:
            float, the cutoff distance
        ethresh:
            float, required energy precision
        box:
            3*3 matrix, box size, a, b, c arranged in rows

    Output:
        kappa:
            float, the attenuation factor
        K1, K2, K3:
            integers, sizes of the k-points mesh
    '''
    kappa = jnp.sqrt(-jnp.log(2*ethresh))/rc
    K1 = jnp.ceil(2 * kappa * box[0, 0] / 3 / ethresh**0.2)
    K2 = jnp.ceil(2 * kappa * box[1, 1] / 3 / ethresh**0.2)
    K3 = jnp.ceil(2 * kappa * box[2, 2] / 3 / ethresh**0.2)

    return kappa, int(K1), int(K2), int(K3)


# @jit_condition(static_argnums=())
def energy_pme(positions, box, pairs,
        Q_local, mScales, pScales, dScales, covalent_map, 
        construct_local_frame_fn, pme_recip_fn, kappa, K1, K2, K3, lmax):
    '''
    This is the top-level wrapper for multipole PME

    Input:
        positions:
            Na * 3: positions
        box: 
            3 * 3: box
        Q_local: 
            Na * (lmax+1)^2: harmonic multipoles of each site in local frame
        mScales, pScale, dScale:
            (Nexcl,): multipole-multipole interaction exclusion scalings: 1-2, 1-3 ...
            for permanent-permanent, permanent-induced, induced-induced interactions
        pairs:
            Np * 2: interacting pair indices
        covalent_map:
            Na * Na: topological distances between atoms, if i, j are topologically distant, then covalent_map[i, j] == 0
        construct_local_frame_fn:
            function: local frame constructors, from generate_local_frame_constructor
        pme_recip:
            function: see recip.py, a reciprocal space calculator
        kappa:
            float: kappa in A^-1
        K1, K2, K3:
            int: max K for reciprocal calculations
        lmax:
            int: maximum L

    Output:
        energy: total pme energy
    '''
    local_frames = construct_local_frame_fn(positions, box)
    Q_global = rot_local2global(Q_local, local_frames, lmax)

    ene_real = pme_real(positions, box, pairs, Q_global, mScales, covalent_map, kappa, lmax)

    ene_recip = pme_recip_fn(positions, box, Q_global)

    ene_self = pme_self(Q_local, kappa, lmax)

    return ene_real + ene_recip + ene_self


# @partial(vmap, in_axes=(0, 0, None, None), out_axes=0)
@jit_condition(static_argnums=(3))
def calc_e_perm(dr, mscales, kappa, lmax=2):

    '''
    This function calculates the ePermCoefs at once
    ePermCoefs is basically the interaction tensor between permanent multipole components
    Everything should be done in the so called quasi-internal (qi) frame
    Energy = \sum_ij qiQI * ePermCoeff_ij * qiQJ

    Inputs:
        dr: 
            float: distance between one pair of particles
        mscales:
            float: scaling factor between permanent - permanent multipole interactions, for each pair
        kappa:
            float: \kappa in PME, unit in A^-1
        lmax:
            int: max L

    Output:
        cc, cd, dd_m0, dd_m1, cq, dq_m0, dq_m1, qq_m0, qq_m1, qq_m2:
            n * 1 array: ePermCoefs
    '''

    # be aware of unit and dimension !!
    rInv = 1 / dr
    rInvVec = jnp.array([DIELECTRIC*(rInv**i) for i in range(0, 9)])
    alphaRVec = jnp.array([(kappa*dr)**i for i in range(0, 10)])
    X = 2 * jnp.exp(-alphaRVec[2]) / jnp.sqrt(np.pi)
    tmp = jnp.array(alphaRVec[1])
    doubleFactorial = 1
    facCount = 1
    erfAlphaR = erf(alphaRVec[1])
        
    # bVec = jnp.empty((6, len(erfAlphaR)))
    bVec = jnp.empty(6)

    bVec = bVec.at[1].set(-erfAlphaR)
    for i in range(2, 6):
        bVec = bVec.at[i].set((bVec[i-1]+(tmp*X/doubleFactorial)))
        facCount += 2
        doubleFactorial *= facCount
        tmp *= 2 * alphaRVec[2]    
    
    # C-C: 1
    cc = rInvVec[1] * (mscales + bVec[2] - alphaRVec[1]*X)
    if lmax >= 1:
        # C-D
        cd = rInvVec[2] * (mscales + bVec[2])
        # D-D: 2
        dd_m0 = -2/3 * rInvVec[3] * (3*(mscales + bVec[3]) + alphaRVec[3]*X)
        dd_m1 = rInvVec[3] * (mscales + bVec[3] - (2/3)*alphaRVec[3]*X)
    else:
        dd_m0 = 0
        dd_m1 = 0

    if lmax >= 2:
        ## C-Q: 1
        cq = (mscales + bVec[3]) * rInvVec[3]
        ## D-Q: 2
        dq_m0 = rInvVec[4] * (3* (mscales + bVec[3]) + (4/3) * alphaRVec[5]*X)
        dq_m1 = -jnp.sqrt(3) * rInvVec[4] * (mscales + bVec[3])
        ## Q-Q
        qq_m0 = rInvVec[5] * (6* (mscales + bVec[4]) + (4/45)* (-3 + 10*alphaRVec[2]) * alphaRVec[5]*X)
        qq_m1 = - (4/15) * rInvVec[5] * (15*(mscales+bVec[4]) + alphaRVec[5]*X)
        qq_m2 = rInvVec[5] * (mscales + bVec[4] - (4/15)*alphaRVec[5]*X)
    else:
        cq = 0
        dq_m0 = 0
        dq_m1 = 0
        qq_m0 = 0
        qq_m1 = 0
        qq_m1 = 0

    return cc, cd, dd_m0, dd_m1, cq, dq_m0, dq_m1, qq_m0, qq_m1, qq_m2


@partial(vmap, in_axes=(0, 0, 0, 0, None, None), out_axes=0)
@jit_condition(static_argnums=(5))
def pme_real_kernel(dr, qiQI, qiQJ, mscales, kappa, lmax=2):
    '''
    This is the heavy-lifting kernel function to compute the realspace multipolar PME 
    Vectorized over interacting pairs

    Input:
        dr: 
            float, the interatomic distances, (np) array if vectorized
        qiQI:
            [(lmax+1)^2] float array, the harmonic multipoles of site i in quasi-internal frame
        qiQJ:
            [(lmax+1)^2] float array, the harmonic multipoles of site j in quasi-internal frame
        mscale:
            float, scaling factor between interacting sites (permanent-permanent)
        kappa:
            float, kappa in unit A^1
        lmax:
            int, maximum angular momentum

    Output:
        energy: 
            float, realspace interaction energy between the sites
    '''

    cc, cd, dd_m0, dd_m1, cq, dq_m0, dq_m1, qq_m0, qq_m1, qq_m2 = calc_e_perm(dr, mscales, kappa, lmax)

    Vij0 = cc*qiQI[0]
    Vji0 = cc*qiQJ[0]

    if lmax >= 1:
        # C-D 
        Vij0 = Vij0 - cd*qiQI[1]
        Vji1 = -cd*qiQJ[0]
        Vij1 = cd*qiQI[0]
        Vji0 = Vji0 + cd*qiQJ[1]
        # D-D m0 
        Vij1 += dd_m0 * qiQI[1]
        Vji1 += dd_m0 * qiQJ[1]    
        # D-D m1 
        Vij2 = dd_m1*qiQI[2]
        Vji2 = dd_m1*qiQJ[2]
        Vij3 = dd_m1*qiQI[3]
        Vji3 = dd_m1*qiQJ[3]

    if lmax >= 2:
        # C-Q
        Vij0 = Vij0 + cq*qiQI[4]
        Vji4 = cq*qiQJ[0]
        Vij4 = cq*qiQI[0]
        Vji0 = Vji0 + cq*qiQJ[4]
        # D-Q m0
        Vij1 += dq_m0*qiQI[4]
        Vji4 += dq_m0*qiQJ[1] 
        # Q-D m0
        Vij4 -= dq_m0*qiQI[1]
        Vji1 -= dq_m0*qiQJ[4]
        # D-Q m1
        Vij2 = Vij2 + dq_m1*qiQI[5]
        Vji5 = dq_m1*qiQJ[2]
        Vij3 += dq_m1*qiQI[6]
        Vji6 = dq_m1*qiQJ[3]
        Vij5 = -(dq_m1*qiQI[2])
        Vji2 += -(dq_m1*qiQJ[5])
        Vij6 = -(dq_m1*qiQI[3])
        Vji3 += -(dq_m1*qiQJ[6])
        # Q-Q m0
        Vij4 += qq_m0*qiQI[4]
        Vji4 += qq_m0*qiQJ[4] 
        # Q-Q m1
        Vij5 += qq_m1*qiQI[5]
        Vji5 += qq_m1*qiQJ[5]
        Vij6 += qq_m1*qiQI[6]
        Vji6 += qq_m1*qiQJ[6]
        # Q-Q m2
        Vij7  = qq_m2*qiQI[7]
        Vji7  = qq_m2*qiQJ[7]
        Vij8  = qq_m2*qiQI[8]
        Vji8  = qq_m2*qiQJ[8]

    if lmax == 0:
        Vij = Vij0
        Vji = Vji0
    elif lmax == 1:
        Vij = jnp.stack((Vij0, Vij1, Vij2, Vij3))
        Vji = jnp.stack((Vji0, Vji1, Vji2, Vji3))
    elif lmax == 2:
        Vij = jnp.stack((Vij0, Vij1, Vij2, Vij3, Vij4, Vij5, Vij6, Vij7, Vij8))
        Vji = jnp.stack((Vji0, Vji1, Vji2, Vji3, Vji4, Vji5, Vji6, Vji7, Vji8))
    else:
        print('Error: Lmax must <= 2')

    return jnp.array(0.5) * (jnp.sum(qiQJ*Vij) + jnp.sum(qiQI*Vji))


# @jit_condition(static_argnums=(7))
def pme_real(positions, box, pairs, 
        Q_global, 
        mScales, covalent_map, 
        kappa, lmax):
    '''
    This is the real space PME calculate function
    NOTE: only deals with permanent-permanent multipole interactions
    It expands the pairwise parameters, and then invoke pme_real_kernel
    It seems pointless to jit it:
    1. the heavy-lifting kernel function is jitted and vmapped
    2. len(pairs) keeps changing throughout the simulation, the function would just recompile everytime

    Input:
        positions:
            Na * 3: positions
        box:
            3 * 3: box, axes arranged in row
        pairs:
            Np * 2: interacting pair indices
        Q_global:
            Na * (l+1)**2: harmonics multipoles of each atom, in global frame
        mScales:
            (Nexcl,): permanent multipole-multipole interaction exclusion scalings: 1-2, 1-3 ...
        covalent_map:
            Na * Na: topological distances between atoms, if i, j are topologically distant, then covalent_map[i, j] == 0
        kappa:
            float: kappa in A^-1
        lmax:
            int: maximum L

    Output:
        ene: pme realspace energy
    '''

    # expand pairwise parameters, from atomic parameters
    pairs = pairs[pairs[:, 0] < pairs[:, 1]]
    box_inv = jnp.linalg.inv(box)
    r1 = positions[pairs[:, 0]]
    r2 = positions[pairs[:, 1]]
    Q_extendi = Q_global[pairs[:, 0]]
    Q_extendj = Q_global[pairs[:, 1]]
    nbonds = covalent_map[pairs[:, 0], pairs[:, 1]]
    mscales = mScales[nbonds-1]

    # deals with geometries
    dr = r1 - r2
    dr = v_pbc_shift(dr, box, box_inv)
    norm_dr = jnp.linalg.norm(dr, axis=-1)
    Ri = build_quasi_internal(r1, r2, dr, norm_dr)
    qiQI = rot_global2local(Q_extendi, Ri, lmax)
    qiQJ = rot_global2local(Q_extendj, Ri, lmax)
    # everything should be pair-specific now
    ene = jnp.sum(pme_real_kernel(norm_dr, qiQI, qiQJ, mscales, kappa, lmax))

    return ene


@jit_condition(static_argnums=(2))
def pme_self(Q_h, kappa, lmax=2):
    '''
    This function calculates the PME self energy

    Inputs:
        Q:
            Na * (lmax+1)^2: harmonic multipoles, local or global does not matter
        kappa:
            float: kappa used in PME

    Output:
        ene_self:
            float: the self energy
    '''
    n_harms = (lmax + 1) ** 2    
    l_list = np.array([0] + [1,]*3 + [2,]*5)[:n_harms]
    l_fac2 = np.array([1] + [3,]*3 + [15,]*5)[:n_harms]
    factor = kappa/np.sqrt(np.pi) * (2*kappa**2)**l_list / l_fac2
    return - jnp.sum(factor[np.newaxis] * Q_h**2) * DIELECTRIC


# below is the validation code
if __name__ == '__main__':
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

    lmax = 2


    # Finish data preparation
    # -------------------------------------------------------------------------------------
    # kappa, K1, K2, K3 = setup_ewald_parameters(rc, ethresh, box)
    # # for debugging
    # kappa = 0.657065221219616
    # construct_local_frames_fn = generate_construct_local_frames(axis_type, axis_indices)
    # energy_force_pme = value_and_grad(energy_pme)
    # e, f = energy_force_pme(positions, box, pairs, Q_local, mScales, pScales, dScales, covalent_map, construct_local_frames_fn, kappa, K1, K2, K3, lmax)
    # print('ok')
    # e, f = energy_force_pme(positions, box, pairs, Q_local, mScales, pScales, dScales, covalent_map, construct_local_frames_fn, kappa, K1, K2, K3, lmax)
    # print(e)

    pme_force = ADMPPmeForce(box, axis_type, axis_indices, covalent_map, rc, ethresh, lmax)
    pme_force.update_env('kappa', 0.657065221219616)

    E, F = pme_force.get_forces(positions, box, pairs, Q_local, mScales, pScales, dScales)
    print('ok')
    E, F = pme_force.get_forces(positions, box, pairs, Q_local, mScales, pScales, dScales)
    print(E)
