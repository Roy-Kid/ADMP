#!/usr/bin/env python3
import numpy as np
import jax.numpy as jnp
import jax.scipy as jsp
from jax_md import partition
from math import factorial
from parser import read_pdb, read_xml, init_residues, assemble_covalent
from jax.config import config
from demo import *
from jax import jit, value_and_grad
import jax
config.update("jax_enable_x64", True)

def dispersion_real(positions,C_list, kappa,mScales):
    '''
    This function calculates the dispersion self energy

    Inputs:
        C_list:
            3 x N_a dispersion susceptibilities
        kappa:
            float: kappa used in dispersion real space

    Outputs:
        ene_real
            float: the real space energy
    '''
    neighbor = neighbor_list_fn(positions)

    # --- build pair stack in numpy --- (copied)
    neighboridx = np.asarray(jax.device_get(neighbor.idx))
    mask = neighboridx != positions.shape[0]
    r = np.sum(mask, axis=1)
    pair1 = np.repeat(np.arange(neighboridx.shape[0]), r)
    pair2 = neighboridx[mask]
    pairs = np.stack((pair1, pair2), axis=1)
    pairs = pairs[pairs[:, 0]<pairs[:, 1]]
    r1 = positions[pairs[:, 0]]
    r2 = positions[pairs[:, 1]]
    d = vmap(displacement_fn)
    dr = d(r1, r2)
    norm_dr = jnp.linalg.norm(dr, axis=1)
    
    nbonds = covalent_map[pairs[:, 0], pairs[:, 1]]
    mscales = mScales[nbonds-1]
      
    #--- g6,g8,g10 C6,C8,C10---
    x = kappa*norm_dr
    g6 = (1+x**2+0.5*x**4)*jnp.exp(-x**2)
    g8 = (1+x**2+0.5*x**4+x**6/6)*jnp.exp(-x**2)
    g10 = (1+x**2+0.5*x**4+x**6/6+x**8/24)*jnp.exp(-x**2)

    C6 = C_list[:,0]
    C8 = C_list[:,1]
    C10 = C_list[:,2]

    #---------------------

    #C66
    C6i = C6[pairs[:,0]]
    C6j = C6[pairs[:,1]]
    C8i = C8[pairs[:,0]]
    C8j = C8[pairs[:,1]]
    C10i= C10[pairs[:,0]]
    C10j = C10[pairs[:,1]]
    disp6 = mscales*g6*C6i*C6j/norm_dr**6 
    disp8 = mscales*g8*C8i*C8j/norm_dr**8
    disp10 = mscales*g10*C10i*C10j/norm_dr**10
    e = sum(disp6)+sum(disp8)+sum(disp10)
    return dr, norm_dr,e




    #g_6 = ()
    #g_8 = ()
    #g_10 = ()
    #ene_real

def dispersion_self(C_list, kappa):
    '''
    This function calculates the dispersion self energy

    Inputs:
        C_list:
            3 x N_a dispersion susceptibilities C_6, C_8, C_10
        kappa:
            float: kappa used in dispersion

    Output:
        ene_self:
            float: the self energy
    '''
    E_6 = - kappa**6 /6 / factorial(6 //2 + 1) * jnp.sum(C_list[0]**2)
    E_8 = - kappa**8 /8 / factorial(8 //2 + 1) * jnp.sum(C_list[1]**2)
    E_10 = - kappa**10 /10 / factorial(10 //2 + 1) * jnp.sum(C_list[2]**2)
    return (E_6 + E_8 + E_10)

def dispersion_reciprocal(positions, box, C_list, kappa, K1, K2, K3):
    '''
    This function calculates the dispersion reciprocal space energy

    Inputs:
        positions:
            N_a x 3 atomic positions
        box:
            3 x 3 box vectors in angstrom
        C_list:
            3 x N_a dispersion susceptibilities C_6, C_8, C_10
        kappa:
            characteristic reciprocal length scale
        K1, K2, K3:
            int: the dimensions of the mesh grid

    Output:
        ene_reciprocal:
            float: the reciprocal space energy
    '''
    N = np.array([K1,K2,K3])
    bspline_range = jnp.arange(-3, 3)
    shifts = jnp.array(jnp.meshgrid(bspline_range, bspline_range, bspline_range)).T.reshape((1, 216, 3))
    
    def get_recip_vectors(N, box):
        """
        Computes reciprocal lattice vectors of the grid
        
        Input:
            N:
                (3,)-shaped array
            box:
                3 x 3 matrix, box parallelepiped vectors arranged in TODO rows or columns?
                
        Output: 
            Nj_Aji_star:
                3 x 3 matrix, the first index denotes reciprocal lattice vector, the second index is the component xyz.
                (lattice vectors arranged in rows)
        """
        Nj_Aji_star = (N.reshape((1, 3)) * jnp.linalg.inv(box)).T
        return Nj_Aji_star

    def u_reference(R_a, Nj_Aji_star):
        """
        Each atom is meshed to dispersion_ORDER**3 points on the m-meshgrid. This function computes the xyz-index of the reference point, which is the point on the meshgrid just above atomic coordinates, and the corresponding values of xyz fractional displacements from real coordinate to the reference point. 
        
        Inputs:
            R_a:
                N_a * 3 matrix containing positions of sites
            Nj_Aji_star:
                3 x 3 matrix, the first index denotes reciprocal lattice vector, the second index is the component xyz.
                (lattice vectors arranged in rows)
                
        Outputs:
            m_u0: 
                N_a * 3 matrix, positions of the reference points of R_a on the m-meshgrid
            u0: 
                N_a * 3 matrix, (R_a - R_m)*a_star values
        """
        
        R_in_m_basis =  jnp.einsum("ij,kj->ki", Nj_Aji_star, R_a)
        
        m_u0 = jnp.ceil(R_in_m_basis).astype(int)
        
        u0 = (m_u0 - R_in_m_basis) + 6/2
        return m_u0, u0

    def bspline6(u):
        """
        Computes the cardinal B-spline function
        """
        return jnp.piecewise(u, 
                            [jnp.logical_and(u>=0, u<1.), 
                            jnp.logical_and(u>=1, u<2.), 
                            jnp.logical_and(u>=2, u<3.), 
                            jnp.logical_and(u>=3, u<4.), 
                            jnp.logical_and(u>=4, u<5.), 
                            jnp.logical_and(u>=5, u<6.)],
                            [lambda u: u**5/120,
                            lambda u: u**5/120 - (u - 1)**5/20,
                            lambda u: u**5/120 + (u - 2)**5/8 - (u - 1)**5/20,
                            lambda u: u**5/120 - (u - 3)**5/6 + (u - 2)**5/8 - (u - 1)**5/20,
                            lambda u: u**5/24 - u**4 + 19*u**3/2 - 89*u**2/2 + 409*u/4 - 1829/20,
                            lambda u: -u**5/120 + u**4/4 - 3*u**3 + 18*u**2 - 54*u + 324/5] )

    def theta_eval(u, M_u):
        """
        Evaluates the value of theta given 3D u values at ... points 
        
        Input:
            u:
                ... x 3 matrix

        Output:
            theta:
                ... matrix
        """
        theta = jnp.prod(M_u, axis = -1)
        return theta

    def mesh_function(u0):
        '''
        Find out the value of the Cardinal B-Spline function near an array of reference points u0

        Inputs:
            u0: 
                a N_a * 3 matrix containing all positions

        Output: 
            theta: 
                a Na * (6**3) matrix, theta evaluated at 6*6*6 integer points about reference points m_u0 
        '''

        N_a = u0.shape[0]
        u = (u0[:, jnp.newaxis, :] + shifts).reshape((N_a*216, 3)) 
        M_u = bspline6(u)
        theta = theta_eval(u, M_u)
        return theta.reshape(N_a, 216)
        
    def C_m_peratom(C, sph_harms):
        """
        Computes <R_t|Q>. See eq. (49) of https://doi.org/10.1021/ct5007983
        
        Inputs:
            C: 
                (N_a, ) matrix containing global frame multipole moments up to lmax,
            sph_harms:
                N_a, 216
        
        Output:
            C_m_pera:
                N_a * 216 matrix, values of theta evaluated on a 6 * 6 block about the atoms
        """
        
        C_m_pera = C[:, jnp.newaxis]*sph_harms

        return C_m_pera
    
    def C_mesh_on_m(C_mesh_pera, m_u0, N):
        """
        spreads the particle mesh onto the grid
        
        Input:
            C_mesh_pera, m_u0, N
            
        Output:
            C_mesh: 
                Nx * Ny * Nz matrix
        """
        indices_arr = jnp.mod(m_u0[:,np.newaxis,:]+shifts, N[np.newaxis, np.newaxis, :])
        
        ### jax trick implementation without using for loop
        C_mesh = jnp.zeros((N[0], N[1], N[2]))
        C_mesh = C_mesh.at[indices_arr[:, :, 0], indices_arr[:, :, 1], indices_arr[:, :, 2]].add(C_mesh_pera)
        
        return C_mesh

    def setup_kpts_integer(N):
        """
        Outputs:
            kpts_int:
                n_k * 3 matrix, n_k = N[0] * N[1] * N[2]
        """
        N_half = N.reshape(3)

        kx, ky, kz = [jnp.roll(jnp.arange(- (N_half[i] - 1) // 2, (N_half[i] + 1) // 2 ), - (N_half[i] - 1) // 2) for i in range(3)]

        kpts_int = jnp.hstack([ki.flatten()[:,jnp.newaxis] for ki in jnp.meshgrid(kz, kx, ky)])

        return kpts_int 

    def setup_kpts(box, kpts_int):
        '''
        This function sets up the k-points used for reciprocal space calculations
        
        Input:
            box:
                3 * 3, three axis arranged in rows
            kpts_int:
                n_k * 3 matrix

        Output:
            kpts:
                4 * K, K=K1*K2*K3, contains kx, ky, kz, k^2 for each kpoint
        '''
        # in this array, a*, b*, c* (without 2*pi) are arranged in column
        box_inv = jnp.linalg.inv(box)

        # K * 3, coordinate in reciprocal space
        kpts = 2 * jnp.pi * kpts_int.dot(box_inv)

        ksr = jnp.sum(kpts**2, axis=1)

        # 4 * K
        kpts = jnp.hstack((kpts, ksr[:, jnp.newaxis])).T

        return kpts
    
    def f_p(x):
        
        f_6 = (1/3 - 2/3*x**2) * jnp.exp(-x**2) + 2/3*x**3*jnp.sqrt(jnp.pi) * jsp.special.erf(x)
        f_8 = 2/(8 - 3)/factorial(8 //2 + 1) * jnp.exp(-x**2) - 4*x**2/(8 - 2)(8-3) * f_6
        f_10 = 2/(10 - 3)/factorial(10 //2 + 1) * jnp.exp(-x**2) - 4*x**2/(10 - 2)(10 - 3) * f_8
        
        return f_6, f_8, f_10

    def E_recip_on_grid(C_mesh_list, box, N, kappa):
        """
        Computes the reciprocal part energy
        """
        N = N.reshape(1,1,3)
        kpts_int = setup_kpts_integer(N)
        kpts = setup_kpts(box, kpts_int)

        m = jnp.linspace(-2,2,5).reshape(5, 1, 1)
        # theta_k : array of shape n_k
        theta_k_sr = jnp.prod(
            jnp.sum(
                bspline6(m + 6/2) * jnp.cos(2*jnp.pi*m*kpts_int[jnp.newaxis] / N),
                axis = 0
            ),
            axis = 1
        )**2

        S_k_6, S_k_8, S_k_10 = [jnp.fft.fftn(C_mesh).flatten() for C_mesh in C_mesh_list]
        
        f_6, f_8, f_10 = f_p(kpts[3]/2/kappa)
        Volume = jnp.linalg.det(box)

        factor = jnp.pi**(3/2)/2/Volume
        E_6 = factor*jnp.sum(kappa**(6-3)*f_6*jnp.abs(S_k_6)**2/theta_k_sr)
        E_8 = factor*jnp.sum(kappa**(8-3)*f_8*jnp.abs(S_k_8)**2/theta_k_sr)
        E_10 = factor*jnp.sum(kappa**(10-3)*f_10*jnp.abs(S_k_10)**2/theta_k_sr)
        return E_6 + E_8 + E_10
    
    Nj_Aji_star = get_recip_vectors(N, box)
    m_u0, u0    = u_reference(positions, Nj_Aji_star)
    sph_harms   = mesh_function(u0)
    C_mesh_list = [C_mesh_on_m(C_m_peratom(C, sph_harms), m_u0, N) for C in C_list]
    E_recip     = E_recip_on_grid(C_mesh_list, box, N, kappa)
    
    # Outputs energy
    return E_recip

if __name__ == '__main__':
    pdb = '../tests/samples/waterdimer_aligned.pdb'
    xml = '../tests/samples/mpidwater.xml'

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
    jitted_pme_reci_energy_and_force = jit(value_and_grad(gen_pme_reciprocal(axis_type, axis_indices)), static_argnums=(4,5,6,7))
    covalent_map = assemble_covalent(residueDicts, natoms)

    ## TODO: setup kappa
    kappa, K1, K2, K3 = setup_ewald_parameters(rc, ethresh, box)
    kappa = 0.328532611
    dielectric = 1389.35455846 # in e^2/A

    import scipy
    from scipy.stats import special_ortho_group

    scipy.random.seed(1000)
    R1 = special_ortho_group.rvs(3)
    R2 = special_ortho_group.rvs(3)

     
    #positions = positions.at[0:3].set(positions[0:3].dot(R1))
    #positions = positions.at[3:6].set(positions[3:6].dot(R2))
    #positions = positions.at[3:].set(positions[3:]+np.array([3.0, 0.0, 0.0]))
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
    #C_list = np.random.rand(natoms,3) # C6 C8 C01 
    C_list = jnp.array([[0.20708234, 0.74246953, 0.39215413],
       [0.18225652, 0.74353941, 0.06958208],
       [0.8853372 , 0.9526444 , 0.93114343],
       [0.41543095, 0.02898166, 0.98202748],
       [0.33963768, 0.70668719, 0.36187707],
       [0.0351059 , 0.85505825, 0.65725351]])

    dr, norm_r,disp_real = dispersion_real(positions,C_list,kappa,mScales)
    disp_self = dispersion_self(C_list,kappa)
    #disp_reciprocal = dispersion_reciprocal(positions, box, C_list, kappa, K1, K2, K3)
    print(disp_self+disp_real)

    
