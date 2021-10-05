import numpy as np
import sys

def sph_harmonics(r, lmax):
    '''
    Find out the value of spherical harmonics, assume the order is:
    00, 10, 11c, 11s, 20, 21c, 21s, 22c, 22s, ...
    Currently supports lmax <= 2

    Input:
         r: 
             a n * 3 matrix containing all positions
         lmax: 
             the maximum l value

    Output: 
         harmonics: 
             a n * (l+1)^2 matrix, the harmonic values of each vector
    '''
    if lmax > 2:
        sys.exit('l > 2 (beyond quadrupole) not supported')
    n_harm = (lmax + 1)**2
    n_sites = len(r)
    harmonics = np.zeros((n_sites, n_harm))

    harmonics[:, 0] = 1
    if lmax >= 1:
        harmonics[:, 1] = r[:, 2]
        harmonics[:, 2] = r[:, 0]
        harmonics[:, 3] = r[:, 1]
    if lmax >= 2:
        rt3 = np.sqrt(3)
        rsq = r**2
        rnorm2 = np.sum(rsq, axis=1)
        harmonics[:, 4] = (3*rsq[:,2] - rnorm2) / 2
        harmonics[:, 5] = rt3 * r[:,0] * r[:,2]
        harmonics[:, 6] = rt3 * r[:,1] * r[:,2]
        harmonics[:, 7] = rt3/2 * (rsq[:,0] - rsq[:,1])
        harmonics[:, 8] = rt3 * r[:,0] * r[:,1]

    return harmonics

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