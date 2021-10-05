import numpy as np
import jax.numpy as jnp
import jax.scipy as jsp

from math import factorial

from jax.config import config
config.update("jax_enable_x64", True)

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