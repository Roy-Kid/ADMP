#!/usr/bin/env python3
# author: yl
# date : 2021-12-16
# version: 0.0.1
import sys
import time

import jax
import jax.numpy as jnp
jnp.set_printoptions(precision=16, suppress=True)
import jax.scipy as jsp
import numpy as np
from jax import grad, jit, value_and_grad, vmap
from jax._src.numpy.lax_numpy import reciprocal
from jax.config import config
from jax_md import partition, space

from admp.parser import assemble_covalent, init_residues, read_pdb, read_xml
from admp.spatial import *
from admp.multipole import *
from admp.pme import *

config.update("jax_enable_x64", True)



def _pme_real(dr, qiQI, qiQJ, kappa, mscales):


    start_ePerm = time.time()
    cc, cd, dd_m0, dd_m1, cq, dq_m0, dq_m1, qq_m0, qq_m1, qq_m2 = calc_e_perm(dr, mscales, kappa)
    end_ePerm = time.time()
    print(f'\t\tePerm costs: {end_ePerm-start_ePerm}')
    Vij0 = cc*qiQJ[:, 0]

    start_mult = time.time()

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

        
    Vij = jnp.stack((Vij0, Vij1, Vij2, Vij3, Vij4, Vij5, Vij6, Vij7, Vij8), axis=1)
    Vji = jnp.stack((Vji0, Vji1, Vji2, Vji3, Vji4, Vji5, Vji6, Vji7, Vji8), axis=1)

    end_mult = time.time()
    print(f'\t\tmulti costs: {end_mult-start_mult}')
    return jnp.array(0.5)*(jnp.sum(qiQI*Vij)+jnp.sum(qiQJ*Vji))

def pme_self(Q_h, kappa, lmax = 2):
    '''
    This function calculates the PME self energy

    Inputs:
        Q:
            N * (lmax+1)^2: harmonic multipoles, local or global does not matter
        kappa:
            float: kappa used in PME

    Output:
        ene_self:
            float: the self energy
    '''
    n_harms = (lmax + 1) ** 2    
    l_list = np.array([0]+[1,]*3+[2,]*5)[:n_harms]
    l_fac2 = np.array([1]+[3,]*3+[15,]*5)[:n_harms]
    factor = kappa/np.sqrt(np.pi) * (2*kappa**2)**l_list / l_fac2
    return - jnp.sum(factor[np.newaxis] * Q_h**2) * dielectric

def gen_pme_reciprocal(axis_type, axis_indices):
    construct_localframes = generate_construct_local_frames(axis_type, axis_indices)
    def pme_reciprocal_on_Qgh(positions, box, Q, kappa, lmax, K1, K2, K3):
        '''
        This function calculates the PME reciprocal space energy

        Inputs:
            positions:
                N_a x 3 atomic positions
            box:
                3 x 3 box vectors in angstrom
            Q:
                N_a x (lmax + 1)**2 multipole values in global frame
            kappa:
                characteristic reciprocal length scale
            lmax:
                maximum value of the multipole level
            K1, K2, K3:
                 int: the dimensions of the mesh grid

        Output:
            ene_reciprocal:
                float: the reciprocal space energy
        '''
        N = np.array([K1,K2,K3])
        ################
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
            Each atom is meshed to PME_ORDER**3 points on the m-meshgrid. This function computes the xyz-index of the reference point, which is the point on the meshgrid just above atomic coordinates, and the corresponding values of xyz fractional displacements from real coordinate to the reference point. 
            
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

        def bspline6prime(u):
            """
            Computes first derivative of the cardinal B-spline function
            """
            return jnp.piecewise(u, 
                                [jnp.logical_and(u>=0., u<1.), 
                                jnp.logical_and(u>=1., u<2.), 
                                jnp.logical_and(u>=2., u<3.), 
                                jnp.logical_and(u>=3., u<4.), 
                                jnp.logical_and(u>=4., u<5.), 
                                jnp.logical_and(u>=5., u<6.)],
                                [lambda u: u**4/24,
                                lambda u: u**4/24 - (u - 1)**4/4,
                                lambda u: u**4/24 + 5*(u - 2)**4/8 - (u - 1)**4/4,
                                lambda u: -5*u**4/12 + 6*u**3 - 63*u**2/2 + 71*u - 231/4,
                                lambda u: 5*u**4/24 - 4*u**3 + 57*u**2/2 - 89*u + 409/4,
                                lambda u: -u**4/24 + u**3 - 9*u**2 + 36*u - 54] )

        def bspline6prime2(u):
            """
            Computes second derivate of the cardinal B-spline function
            """
            return jnp.piecewise(u, 
                                [jnp.logical_and(u>=0., u<1.), 
                                jnp.logical_and(u>=1., u<2.), 
                                jnp.logical_and(u>=2., u<3.), 
                                jnp.logical_and(u>=3., u<4.), 
                                jnp.logical_and(u>=4., u<5.), 
                                jnp.logical_and(u>=5., u<6.)],
                                [lambda u: u**3/6,
                                lambda u: u**3/6 - (u - 1)**3,
                                lambda u: 5*u**3/3 - 12*u**2 + 27*u - 19,
                                lambda u: -5*u**3/3 + 18*u**2 - 63*u + 71,
                                lambda u: 5*u**3/6 - 12*u**2 + 57*u - 89,
                                lambda u: -u**3/6 + 3*u**2 - 18*u + 36,] )


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

        def thetaprime_eval(u, Nj_Aji_star, M_u, Mprime_u):
            """
            First derivative of theta with respect to x,y,z directions
            
            Input:
                u
                Nj_Aji_star:
                    reciprocal lattice vectors
            
            Output:
                N_a * 3 matrix
            """

            div = jnp.array([
                Mprime_u[:, 0] * M_u[:, 1] * M_u[:, 2],
                Mprime_u[:, 1] * M_u[:, 2] * M_u[:, 0],
                Mprime_u[:, 2] * M_u[:, 0] * M_u[:, 1],
            ]).T
            
            # Notice that u = m_u0 - R_in_m_basis + 6/2
            # therefore the Jacobian du_j/dx_i = - Nj_Aji_star
            return jnp.einsum("ij,kj->ki", -Nj_Aji_star, div)

        def theta2prime_eval(u, Nj_Aji_star, M_u, Mprime_u, M2prime_u):
            """
            compute the 3 x 3 second derivatives of theta with respect to xyz
            
            Input:
                u
                Nj_Aji_star
            
            Output:
                N_A * 3 * 3
            """

            div_00 = M2prime_u[:, 0] * M_u[:, 1] * M_u[:, 2]
            div_11 = M2prime_u[:, 1] * M_u[:, 0] * M_u[:, 2]
            div_22 = M2prime_u[:, 2] * M_u[:, 0] * M_u[:, 1]
            
            div_01 = Mprime_u[:, 0] * Mprime_u[:, 1] * M_u[:, 2]
            div_02 = Mprime_u[:, 0] * Mprime_u[:, 2] * M_u[:, 1]
            div_12 = Mprime_u[:, 1] * Mprime_u[:, 2] * M_u[:, 0]

            div_10 = div_01
            div_20 = div_02
            div_21 = div_12
            
            div = jnp.array([
                [div_00, div_01, div_02],
                [div_10, div_11, div_12],
                [div_20, div_21, div_22],
            ]).swapaxes(0, 2)
            
            # Notice that u = m_u0 - R_in_m_basis + 6/2
            # therefore the Jacobian du_j/dx_i = - Nj_Aji_star
            return jnp.einsum("im,jn,kmn->kij", -Nj_Aji_star, -Nj_Aji_star, div)

        def sph_harmonics_GO(u0, Nj_Aji_star):
            '''
            Find out the value of spherical harmonics GRADIENT OPERATORS, assume the order is:
            00, 10, 11c, 11s, 20, 21c, 21s, 22c, 22s, ...
            Currently supports lmax <= 2

            Inputs:
                u0: 
                    a N_a * 3 matrix containing all positions
                Nj_Aji_star:
                    reciprocal lattice vectors in the m-grid

            Output: 
                harmonics: 
                    a Na * (6**3) * (l+1)^2 matrix, STGO operated on theta,
                    evaluated at 6*6*6 integer points about reference points m_u0 
            '''
            
            n_harm = (lmax + 1)**2

            N_a = u0.shape[0]
            u = (u0[:, jnp.newaxis, :] + shifts).reshape((N_a*216, 3)) 

            M_u = bspline6(u)
            theta = theta_eval(u, M_u)
            if lmax == 0:
                return theta.reshape(N_a, 216, n_harm)
            
            # dipole
            Mprime_u = bspline6prime(u)
            thetaprime = thetaprime_eval(u, Nj_Aji_star, M_u, Mprime_u)
            harmonics_1 = jnp.stack(
                [theta,
                thetaprime[:, 2],
                thetaprime[:, 0],
                thetaprime[:, 1]],
                axis = -1
            )
            
            if lmax == 1:
                return harmonics_1.reshape(N_a, 216, n_harm)

            # quadrapole
            M2prime_u = bspline6prime2(u)
            theta2prime = theta2prime_eval(u, Nj_Aji_star, M_u, Mprime_u, M2prime_u)
            rt3 = jnp.sqrt(3)
            harmonics_2 = jnp.hstack(
                [harmonics_1,
                jnp.stack([(3*theta2prime[:,2,2] - jnp.trace(theta2prime, axis1=1, axis2=2)) / 2,
                rt3 * theta2prime[:, 0, 2],
                rt3 * theta2prime[:, 1, 2],
                rt3/2 * (theta2prime[:, 0, 0] - theta2prime[:, 1, 1]),
                rt3 * theta2prime[:, 0, 1]], axis = 1)]
            )
            if lmax == 2:
                return harmonics_2.reshape(N_a, 216, n_harm)
            else:
                raise NotImplementedError('l > 2 (beyond quadrupole) not supported')
            
        def Q_m_peratom(Q, sph_harms):
            """
            Computes <R_t|Q>. See eq. (49) of https://doi.org/10.1021/ct5007983
            
            Inputs:
                Q: 
                    N_a * (l+1)**2 matrix containing global frame multipole moments up to lmax,
                sph_harms:
                    N_a, 216, (l+1)**2
            
            Output:
                Q_m_pera:
                    N_a * 216 matrix, values of theta evaluated on a 6 * 6 block about the atoms
            """
            
            N_a = sph_harms.shape[0]
            
            if lmax > 2:
                raise NotImplementedError('l > 2 (beyond quadrupole) not supported')

            Q_dbf = Q[:, 0]
            if lmax >= 1:
                Q_dbf = jnp.hstack([Q_dbf[:,jnp.newaxis], Q[:,1:4]])
            if lmax >= 2:
                Q_dbf = jnp.hstack([Q_dbf, Q[:,4:9]/3])
            
            Q_m_pera = jnp.sum( Q_dbf[:,jnp.newaxis,:]* sph_harms, axis=2)

            assert Q_m_pera.shape == (N_a, 216)
            return Q_m_pera
        
        def Q_mesh_on_m(Q_mesh_pera, m_u0, N):
            """
            spreads the particle mesh onto the grid
            
            Input:
                Q_mesh_pera, m_u0, N
                
            Output:
                Q_mesh: 
                    Nx * Ny * Nz matrix
            """


            indices_arr = jnp.mod(m_u0[:,np.newaxis,:]+shifts, N[np.newaxis, np.newaxis, :])
            
            ### jax trick implementation without using for loop
            ### NOTICE: this implementation does not work with numpy!
            Q_mesh = jnp.zeros((N[0], N[1], N[2]))
            Q_mesh = Q_mesh.at[indices_arr[:, :, 0], indices_arr[:, :, 1], indices_arr[:, :, 2]].add(Q_mesh_pera)
            
            return Q_mesh

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

        def E_recip_on_grid(Q_mesh, box, N, kappa):
            """
            Computes the reciprocal part energy
            """
            
            N = N.reshape(1,1,3)
            kpts_int = setup_kpts_integer(N)
            kpts = setup_kpts(box, kpts_int)

            m = jnp.linspace(-2,2,5).reshape(5, 1, 1)
            # theta_k : array of shape n_k
            theta_k = jnp.prod(
                jnp.sum(
                    bspline6(m + 6/2) * jnp.cos(2*jnp.pi*m*kpts_int[jnp.newaxis] / N),
                    axis = 0
                ),
                axis = 1
            )

            S_k = jnp.fft.fftn(Q_mesh)

            S_k = S_k.flatten()

            E_k = 2*jnp.pi/kpts[3,1:]/jnp.linalg.det(box) * jnp.exp( - kpts[3, 1:] /4 /kappa**2) * jnp.abs(S_k[1:]/theta_k[1:])**2
            return jnp.sum(E_k)
        
        Nj_Aji_star = get_recip_vectors(N, box)
        m_u0, u0    = u_reference(positions, Nj_Aji_star)
        sph_harms   = sph_harmonics_GO(u0, Nj_Aji_star)
        Q_mesh_pera = Q_m_peratom(Q, sph_harms)
        Q_mesh      = Q_mesh_on_m(Q_mesh_pera, m_u0, N)
        E_recip     = E_recip_on_grid(Q_mesh, box, N, kappa)
        
        # Outputs energy in OPENMM units
        return E_recip*dielectric

    def pme_reciprocal(positions, box,  Q_lh, kappa, lmax, K1, K2, K3):
        localframes = construct_localframes(positions, box)
        Q_gh = rot_local2global(Q_lh, localframes, lmax)
        return pme_reciprocal_on_Qgh(positions, box, Q_gh, kappa, lmax, K1, K2, K3)
    return pme_reciprocal

def calc_distance(r1, r2):
    d = vmap(displacement_fn)
    dr = d(r1, r2)
    norm_dr = jnp.linalg.norm(dr, axis=1)
    return dr, norm_dr

def build_quasi_internal(r1, r2, dr, norm_dr):

    vectorZ = dr/norm_dr.reshape((-1, 1))

    vectorX = jnp.where(jnp.logical_or(r1[:, 1] != r2[:, 1], r1[:, 2]!=r2[:, 2]).reshape((-1, 1)), vectorZ+jnp.array([1., 0., 0.]), vectorZ + jnp.array([0., 1., 0.]))
    dot = jnp.sum(vectorZ * vectorX, axis=1)
    vectorX -= vectorZ * dot[:, jnp.newaxis]
    vectorX = vectorX / jnp.linalg.norm(vectorX, axis=1).reshape((-1, 1))
    vectorY = jnp.cross(vectorZ, vectorX)
    return jnp.stack([vectorX, vectorY, vectorZ], axis=1)

def build_a_quasi_internal(pos, positions, dr, pairidx):
    
    r1 = pos
    r2 = positions[pairidx]
    norm_dr = jnp.linalg.norm(dr)
        
    vectorZ = dr/norm_dr
    
    vectorX = jnp.where(jnp.logical_or(r1[1]!=r2[1], r1[2]!=r2[2]), vectorZ+jnp.array([1., 0., 0.]), vectorZ + jnp.array([0., 1., 0.]))
    dot = vectorZ * vectorX
    vectorX -= vectorZ * dot
    vectorX /= jnp.linalg.norm(vectorX)
    vectorY = jnp.cross(vectorZ, vectorX)
    return jnp.stack([vectorX, vectorY, vectorZ], axis=1)

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


    # --- build pair stack in numpy --- (copied)
    neighboridx = jax.device_get(nbr.idx)
    mask = neighboridx != positions.shape[0]
    r = np.sum(mask, axis=1)
    pair1 = np.repeat(np.arange(neighboridx.shape[0]), r)
    pair2 = neighboridx[mask]
    pairs = np.stack((pair1, pair2), axis=1)
    pairs = pairs[pairs[:, 0]<pairs[:, 1]]
    r1 = positions[pairs[:, 0]]
    r2 = positions[pairs[:, 1]]
    d = jit(vmap(displacement_fn))
    dr = d(r1, r2)
    norm_dr = jnp.linalg.norm(dr, axis=1)

    nbonds = covalent_map[pairs[:, 0], pairs[:, 1]]
    mscales = mScales[nbonds-1]
    #--- g6,g8,g10 C6,C8,C10---
    x = kappa*norm_dr
    g6 = (1+x**2+0.5*x**4)*jnp.exp(-x**2)
    g8 = (1+x**2+0.5*x**4+x**6/6)*jnp.exp(-x**2)
    g10 = (1+x**2+0.5*x**4+x**6/6+x**8/24)*jnp.exp(-x**2)

    C6 = C_list[0]
    C8 = C_list[1]
    C10 = C_list[2]

    #---------------------

    #C66
    C6i = C6[pairs[:,0]]
    C6j = C6[pairs[:,1]]
    C8i = C8[pairs[:,0]]
    C8j = C8[pairs[:,1]]
    C10i= C10[pairs[:,0]]
    C10j = C10[pairs[:,1]]
    disp6 = (mscales+g6-1)*C6i*C6j/norm_dr**6
    disp8 = (mscales+g8-1)*C8i*C8j/norm_dr**8
    disp10 = (mscales+g10-1)*C10i*C10j/norm_dr**10
    e = jnp.sum(disp6)+jnp.sum(disp8)+jnp.sum(disp10)
    return e
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
    E_6 = -kappa**6/12 * jnp.sum(C_list[0]**2)
    E_8 = -kappa**8/48 * jnp.sum(C_list[1]**2)
    E_10 = -kappa**10/240 * jnp.sum(C_list[2]**2)
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
        f_6 = (1/3 - 2/3*x**2) * jnp.exp(-x**2) + 2/3*x**3*jnp.sqrt(jnp.pi) * jsp.special.erfc(x)
        f_8 = 1/15 * jnp.exp(-x**2) - 2*x**2/15 * f_6
        f_10 = 1/84  * jnp.exp(-x**2) - x**2/14 * f_8
        
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
        
        f_6, f_8, f_10 = f_p(jnp.sqrt(kpts[3])/2/kappa)
        Volume = jnp.linalg.det(box)

        factor = jnp.pi**(3/2)/2/Volume
        E_6 = factor*kappa**3*jnp.sum(f_6*jnp.abs(S_k_6)**2/theta_k_sr)
        E_8 = factor*kappa**5*jnp.sum(f_8*jnp.abs(S_k_8)**2/theta_k_sr)
        E_10 = factor*kappa**7*jnp.sum(f_10*jnp.abs(S_k_10)**2/theta_k_sr) 
        return E_6 + E_8 + E_10
    
    Nj_Aji_star = get_recip_vectors(N, box)
    m_u0, u0    = u_reference(positions, Nj_Aji_star)
    sph_harms   = mesh_function(u0)
    C_mesh_list = [C_mesh_on_m(C_m_peratom(C, sph_harms), m_u0, N) for C in C_list]
    E_recip     = E_recip_on_grid(C_mesh_list, box, N, kappa)
    
    # Outputs energy
    return E_recip

jitted_rot_local2global = jit(rot_local2global, static_argnums=2)
jitted_rot_global2local = jit(rot_global2local, static_argnums=2)
jitted_pme_self_energy_and_force = jit(value_and_grad(pme_self), static_argnums=2)
jitted_build_quasi_internal = jit(build_quasi_internal)
jitted_calc_distance = jit(calc_distance)

def real_space(positions, Qlocal, box, kappa):

    # --- convert Qlocal to Qglobal ---
    local_frames = jitted_construct_localframes(positions, box)
    Qglobal = jitted_rot_local2global(Qlocal, local_frames, 2)

    # --- end convert Qlocal to Qglobal ---
    # mask = neighbor.idx != positions.shape[0]
    # pos_neigh = positions[neighbor.idx]
    
    # d = displacement_fn
    # d = vmap(vmap(d, (None, 0)))
    # dR = d(positions, pos_neigh)

    # b = build_a_quasi_internal
    # b = vmap(vmap(b, (None, None, None, 0)), (None, 0, 0, 0))
    # idx = jnp.arange(neighbor.idx.shape[0])
    # Ri = b(positions, dR, idx, neighbor.idx)

    # Q_extendi = Qglobal[neighbor.idx]
    
    # qiQ = jitted_rot_global2local(Q_extendi, Ri, 2)
    # mscales = mScales
    
    # --- build pair stack in numpy ---
    start_pair = time.time()
    neighboridx = jax.device_get(nbr.idx)
    mask = neighboridx != positions.shape[0]
    r = np.sum(mask, axis=1)
    pair1 = np.repeat(np.arange(neighboridx.shape[0]), r)
    pair2 = neighboridx[mask]
    pairs = np.stack((pair1, pair2), axis=1)
    pairs = pairs[pairs[:, 0]<pairs[:, 1]]
    end_pair = time.time()
    print(f'\tpair cost: {end_pair-start_pair}')

    # --- end build pair stack in numpy ---

    # --- build quasi internal in numpy ---
    r1 = positions[pairs[:, 0]]
    r2 = positions[pairs[:, 1]]
    d = jit(vmap(displacement_fn))
    dr = d(r1, r2)
    norm_dr = jnp.linalg.norm(dr, axis=1)
    start_quasi = time.time()
    Ri = jitted_build_quasi_internal(r1, r2, dr, norm_dr)
    end_quasi = time.time()
    print(f'\tquasi: {end_quasi-start_quasi}')
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
    # e, f = jitted_pme_real_energy_and_force(norm_dr, qiQJ, qiQI, kappa, mscales)
    start_calc = time.time()
    e = _pme_real(norm_dr, qiQJ, qiQI, kappa, mscales)
    end_calc = time.time()
    print(f'\tcalc costs: {end_calc - start_calc}')
    return e



if __name__ == '__main__':
    # --- prepare data ---

    # pdb = 'tests/samples/waterdimer_aligned.pdb'
    #pdb = 'tests/samples/waterbox_31ang.pdb'
    pdb = str(sys.argv[1])
    #pdb = f'tests/samples/water1.pdb'
    xml = 'mpidwater.xml'

    # return a dict with raw data from pdb
    pdbinfo = read_pdb(pdb)

    serials = pdbinfo['serials']
    names = pdbinfo['names']
    resNames = pdbinfo['resNames']
    resSeqs = pdbinfo['resSeqs']
    positions = pdbinfo['positions']
    charges = pdbinfo['charges']
    box = pdbinfo['box'] # a, b, c, α, β, γ
    positions = jnp.asarray(positions)
    lx = box[0]
    ly = box[1]
    lz = box[2]
    
    dielectric = 1389.35455846 # in e^2/A
    mScales = jnp.array([0.0, 0.0, 0.0, 1.0, 1.0])
    pScales = jnp.array([0.0, 0.0, 0.0, 1.0, 1.0])
    dScales = jnp.array([0.0, 0.0, 0.0, 1.0, 1.0])

    box = jnp.eye(3)*jnp.array([lx, ly, lz])

    rc = 4  # in Angstrom
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

    Q = jnp.array(Q)
    Qlocal = convert_cart2harm(Q, 2)
    # Qlocal = jnp.array(Qlocal)
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
    kappa, K1, K2, K3 = setup_ewald_parameters(rc, ethresh, box)
    kappa = 0.657065221219616 
    # === pre-compile compute function ===
    jitted_pme_reci_energy_and_force = jit(value_and_grad(gen_pme_reciprocal(axis_type, axis_indices)), static_argnums=(4,5,6,7))
    construct_localframes = generate_construct_local_frames(axis_type, axis_indices)
    jitted_construct_localframes = jit(construct_localframes)
    jitted_pme_real_energy_and_force = jit(value_and_grad(real_space, argnums=0))

    jitted_disp_real_energy_and_force = jit(value_and_grad(dispersion_real))
    jitted_disp_self_energy_and_force = jit(value_and_grad(dispersion_self))
    jitted_disp_reci_energy_and_force = jit(value_and_grad(dispersion_reciprocal),static_argnums=(4,5,6))
    # === prepare neighborlist ===
    
    cell_size = rc
    displacement_fn, shift_fn = space.periodic_general(box, fractional_coordinates=False)
    neighbor_list_fn = partition.neighbor_list(displacement_fn, box, cell_size, 0)
    start_nbl = time.time()
    nbr = neighbor_list_fn.allocate(positions)
    #nbr = nbr.update(positions)


    # == C_list
    C_list = np.random.rand(3,natoms)
    nmol=int(natoms/3)
    for i in range(nmol):
        a = i*3
        b = i*3+1
        c = i*3+2
        C_list[0][a]=37.19677405
        C_list[0][b]=7.6111103
        C_list[0][c]=7.6111103
        C_list[1][a]=85.26810658
        C_list[1][b]=11.90220148
        C_list[1][c]=11.90220148
        C_list[2][a]=134.44874488
        C_list[2][b]=15.05074749
        C_list[2][c]=15.05074749 
    end_nbl = time.time()
    print(f'nbl costs: {end_nbl - start_nbl}')
    # === start to calculate ADMP === 
    ereal, freal = jitted_pme_real_energy_and_force(positions, Qlocal, box, kappa)
    ereci, freci = jitted_pme_reci_energy_and_force(positions, box, Qlocal, kappa, 2, K1, K2, K3)
    eself, fself = jitted_pme_self_energy_and_force(Qlocal, kappa)

    edisp_real, fdisp_real = jitted_disp_real_energy_and_force(positions,C_list,kappa,mScales)
    edisp_self, fdisp_self = jitted_disp_self_energy_and_force(C_list, kappa)    
    edisp_reci, fdisp_reci = jitted_disp_reci_energy_and_force(positions, box, C_list, kappa, K1, K2, K3)
    # --- start calc pme ---
    epoch = 100
    
    #start_real = time.time()
    #for i in range(epoch):
        #ereal, freal = jitted_pme_real_energy_and_force(positions, Qlocal, box, kappa)
        #ereal.block_until_ready()
        #freal.block_until_ready()
    #end_real = time.time()

    #start_reci = time.time()
    #for i in range(epoch):
        #ereci, freci = jitted_pme_reci_energy_and_force(positions, box, Qlocal, kappa, 2, K1, K2, K3)
    #ereci.block_until_ready()
    #freci.block_until_ready()
    #end_reci = time.time()
    
    #start_self = time.time()
    #for i in range(epoch):
        #eself, fself = jitted_pme_self_energy_and_force(Qlocal, kappa)
    #eself.block_until_ready()
    #self.block_until_ready()
    #end_self = time.time()
    
    start_real = time.time()
    for i in range(epoch):
        edisp_real, fdisp_real = jitted_disp_real_energy_and_force(positions,C_list,kappa,mScales)
    edisp_real.block_until_ready()
    fdisp_real.block_until_ready()
    end_real = time.time()

    start_self = time.time()
    for i in range(epoch):
        edisp_self, fdisp_self = jitted_disp_self_energy_and_force(C_list, kappa) 
    edisp_self.block_until_ready()
    fdisp_self.block_until_ready()
    end_self = time.time()

    start_reci = time.time()
    for i in range(epoch):
        edisp_reci ,fdisp_reci = jitted_disp_reci_energy_and_force(positions, box, C_list, kappa, K1, K2, K3)
    edisp_reci.block_until_ready()
    fdisp_reci.block_until_ready()
    end_reci = time.time()
    print(ereal)
    print(ereci)
    print(eself)
    print(f'total : {edisp_real+edisp_reci+edisp_self}')
    print(edisp_real)
    print(edisp_reci)
    print(edisp_self)
    print(f'real e&f cost: {(end_real - start_real)/epoch}')
    print(f'reci e&f cost: {(end_reci - start_reci)/epoch}')
    print(f'self e&f cost: {(end_self - start_self)/epoch}')
    

        
    
    # --- end calc pme ---
    
    # --- start calc dispersion ---
    
    # c_list = [] # 3 x N_a dispersion susceptibilities C_6, C_8, C_10
    
    # eself_dispersion = dispersion_self(c_list, kappa)
    
    # ereci_dispersion = dispersion_reciprocal(positions, box, c_list, K1, K2, K3)
    
    # --- end calc dispersion ---

    # jax.profiler.stop_trace()

    # finite differences
    # findiff = np.empty((6, 3))
    # eps = 1e-4
    # delta = np.zeros((6,3)) 
    # for i in range(6): 
    #   for j in range(3): 
    #     delta[i][j] = eps 
    #     findiff[i][j] = (real_space(positions+delta/2, Qlocal, box, kappa, mScales) - real_space(positions-delta/2, Qlocal, box, kappa, mScales))/eps
    #     delta[i][j] = 0
    # print('partial diff')
    # print(findiff)
    # print(freal)

    
