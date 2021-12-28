#!/usr/bin/env python
import jax
from jax import vmap, value_and_grad
from admp.settings import *
from admp.spatial import *
from admp.pme import *


def energy_disp_pme(positions, box, pairs,
        c_list, mScales, covalent_map,
        kappa, K1, K2, K3, pmax):
    '''
    Top level wrapper for dispersion pme

    Input:
        positions:
            Na * 3: positions
        box:
            3 * 3: box, axes arranged in row
        pairs:
            Np * 2: interacting pair indices
        c_list:
            Na * (pmax-4)/2: atomic dispersion coefficients
        mScales:
            (Nexcl,): permanent multipole-multipole interaction exclusion scalings: 1-2, 1-3 ...
        covalent_map:
            Na * Na: topological distances between atoms, if i, j are topologically distant, then covalent_map[i, j] == 0
        kappa:
            float: kappa in A^-1
        K1, K2, K3:
            int: max K for reciprocal calculations
        pmax:
            int array: maximal exponents (p) to compute, e.g., (6, 8, 10)

    Output:
        energy: total dispersion pme energy
    '''

    ene_real = disp_pme_real(positions, box, pairs, c_list, mScales, covalent_map, kappa, pmax)

    ene_self = disp_pme_self(c_list, kappa, pmax)

    return ene_real


def disp_pme_real(positions, box, pairs, 
        c_list, 
        mScales, covalent_map, 
        kappa, pmax):
    '''
    This function calculates the dispersion real space energy
    It expands the atomic parameters to pairwise parameters

    Input:
        positions:
            Na * 3: positions
        box:
            3 * 3: box, axes arranged in row
        pairs:
            Np * 2: interacting pair indices
        c_list:
            Na * (pmax-4)/2: atomic dispersion coefficients
        mScales:
            (Nexcl,): permanent multipole-multipole interaction exclusion scalings: 1-2, 1-3 ...
        covalent_map:
            Na * Na: topological distances between atoms, if i, j are topologically distant, then covalent_map[i, j] == 0
        kappa:
            float: kappa in A^-1
        pmax:
            int array: maximal exponents (p) to compute, e.g., (6, 8, 10)

    Output:
        ene: dispersion pme realspace energy
    '''

    # expand pairwise parameters
    pairs = pairs[pairs[:, 0] < pairs[:, 1]]

    box_inv = jnp.linalg.inv(box)
    ri = positions[pairs[:, 0]]
    rj = positions[pairs[:, 1]]
    nbonds = covalent_map[pairs[:, 0], pairs[:, 1]]
    mscales = mScales[nbonds-1]

    ci = c_list[pairs[:, 0], :]
    cj = c_list[pairs[:, 1], :]

    ene_real = jnp.sum(disp_pme_real_kernel(ri, rj, ci, cj, box, box_inv, mscales, kappa, pmax))

    return jnp.sum(ene_real)


@partial(vmap, in_axes=(0, 0, 0, 0, None, None, 0, None, None), out_axes=(0))
@jit_condition(static_argnums=(8))
def disp_pme_real_kernel(ri, rj, ci, cj, box, box_inv, mscales, kappa, pmax):
    '''
    The kernel to calculate the realspace dispersion energy
    
    Inputs:
        ri: 
            Np * 3: position i
        rj:
            Np * 3: position j
        ci: 
            Np * (pmax-4)/2: dispersion coeffs of i, c6, c8, c10 etc
        cj:
            Np * (pmax-4)/2: dispersion coeffs of j, c6, c8, c10 etc
        kappa:
            float: kappa
        pmax:
            int: largest p in 1/r^p, assume starting from 6 with increment of 2

    Output:
        energy: 
            float: the dispersion pme energy
    '''
    dr = ri - rj
    dr = pbc_shift(dr, box, box_inv)
    dr2 = jnp.dot(dr, dr)
    x2 = kappa * kappa * dr2
    g = g_p(x2, pmax)
    dr6 = dr2 * dr2 * dr2
    ene = (mscales + g[0] - 1) * ci[0] * cj[0] / dr6
    if pmax >= 8:
        dr8 = dr6 * dr2
        ene += (mscales + g[1] - 1) * ci[1] * cj[1] / dr8
    if pmax >= 10:
        dr10 = dr8 * dr2
        ene += (mscales + g[2] - 1) * ci[2] * cj[2] / dr10
    return ene


def g_p(x2, pmax):
    '''
    Compute the g(x, p) function

    Inputs:
        x:
            float: the input variable
        pmax:
            int: the maximal powers of dispersion, here we assume evenly spacing even powers starting from 6
            e.g., (6,), (6, 8) or (6, 8, 10)

    Outputs:
        g:
            (p-4)//2: g(x, p)
    '''

    x4 = x2 * x2
    x8 = x4 * x4
    exp_x2 = jnp.exp(-x2)
    g6 = 1 + x2 + 0.5*x4
    if pmax >= 8:
        g8 = g6 + x4*x2/6
    if pmax >= 10:
        g10 = g8 + x8/24

    if pmax == 6:
        g = jnp.array([g6])
    elif pmax == 8:
        g = jnp.array([g6, g8])
    elif pmax == 10:
        g = jnp.array([g6, g8, g10])

    return g * exp_x2


@jit_condition(static_argnums=(2))
def disp_pme_self(c_list, kappa, pmax):
    '''
    This function calculates the dispersion self energy

    Inputs:
        c_list:
            Na * 3: dispersion susceptibilities C_6, C_8, C_10
        kappa:
            float: kappa used in dispersion

    Output:
        ene_self:
            float: the self energy
    '''
    E_6 = -kappa**6/12 * jnp.sum(c_list[:, 0]**2)
    if pmax >= 8:
        E_8 = -kappa**8/48 * jnp.sum(c_list[:, 1]**2)
    if pmax >= 10:
        E_10 = -kappa**10/240 * jnp.sum(c_list[:, 2]**2)
    E = E_6
    if pmax >= 8:
        E += E_8
    if pmax >= 10:
        E += E_10
    return E


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

    covalent_map = assemble_covalent(residueDicts, n_atoms)
    displacement_fn, shift_fn = space.periodic_general(box, fractional_coordinates=False)
    neighbor_list_fn = partition.neighbor_list(displacement_fn, box, rc, 0, format=partition.OrderedSparse)
    nbr = neighbor_list_fn.allocate(positions)
    pairs = nbr.idx.T

    pmax = 10
    kappa, K1, K2, K3 = setup_ewald_parameters(rc, ethresh, box)
    kappa = 0.657065221219616

    # construct the C list
    c_list = np.zeros((3,n_atoms))
    nmol=int(n_atoms/3)
    for i in range(nmol):
        a = i*3
        b = i*3+1
        c = i*3+2
        c_list[0][a]=37.19677405
        c_list[0][b]=7.6111103
        c_list[0][c]=7.6111103
        # c_list[1][a]=85.26810658
        # c_list[1][b]=11.90220148
        # c_list[1][c]=11.90220148
        # c_list[2][a]=134.44874488
        # c_list[2][b]=15.05074749
        # c_list[2][c]=15.05074749
    c_list = jnp.array(c_list.T)
    energy_force_disp_pme = value_and_grad(energy_disp_pme)
    e, f = energy_force_disp_pme(positions, box, pairs, c_list, mScales, covalent_map, kappa, K1, K2, K3, pmax)
    print('ok')
    e, f = energy_force_disp_pme(positions, box, pairs, c_list, mScales, covalent_map, kappa, K1, K2, K3, pmax)
    print(e)

