import numpy as np

def pbc_shift(rvecs, box, box_inv):
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
    svecs = rvecs.dot(box_inv)
    svecs -= np.floor(svecs + 0.5)
    return svecs.dot(box)

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

def rot_global2local(Q, R, lmax):
    '''
    This one rotate harmonic moments Q from global frame to local frame

    Input:
        Q: 
            n * (l+1)^2, stores the multipole moments of each site
        R: 
            n * 3 * 3, stores the Rotation matrix for each site, the R is defined as:
            [r1, r2, r3]^T, with r1, r2, r3 being the local frame axes
        lmax:
            integer, the maximum l

    Output:
        Qrot:
            n * (l+1)^2, stores the rotated multipole moments
    '''
    if lmax > 2:
        sys.exit('l > 2 (beyond quadrupole) not supported')

    n_harm = (lmax + 1)**2
    # do not touch monopole
    n_sites = len(Q)

    Qrot = np.zeros((n_sites, n_harm))

    # do not touch the monopoles
    Qrot[:, 0] = Q[:, 0]
    # for dipole
    if lmax >= 1:
        C1 = np.zeros((3, 3))
        C1[0, 1] = 1
        C1[1, 2] = 1
        C1[2, 0] = 1
        # dipoles arrange in row
        dipoles = Q[:, 1:4]
        # the rotation matrix
        R1 = np.einsum('ijk,kl->ijl', np.einsum('jk,ikl->ijl', C1.T, R), C1)
        # rotate
        Qrot[:, 1:4] = np.einsum('ijk,ik->ij', R1, dipoles)
    if lmax >= 2:
        rt3 = np.sqrt(3)
        R_flatten = R.reshape((n_sites, -1))
        ixx, ixy, ixz, iyx, iyy, iyz, izx, izy, izz = range(9)
        xx = R_flatten[:, ixx]
        xy = R_flatten[:, ixy]
        xz = R_flatten[:, ixz]
        yx = R_flatten[:, iyx]
        yy = R_flatten[:, iyy]
        yz = R_flatten[:, iyz]
        zx = R_flatten[:, izx]
        zy = R_flatten[:, izy]
        zz = R_flatten[:, izz]
        quadrupoles = Q[:, 4:9]
        C_gl = np.zeros((n_sites, 5, 5))
        # construct the local->global transformation matrix
        # this is copied directly from the convert_mom_to_xml.py code
        C_gl[:,0,0] = (3*zz**2-1)/2
        C_gl[:,0,1] = rt3*zx*zz
        C_gl[:,0,2] = rt3*zy*zz
        C_gl[:,0,3] = (rt3*(-2*zy**2-zz**2+1))/2
        C_gl[:,0,4] = rt3*zx*zy
        C_gl[:,1,0] = rt3*xz*zz
        C_gl[:,1,1] = 2*xx*zz-yy
        C_gl[:,1,2] = yx+2*xy*zz
        C_gl[:,1,3] = -2*xy*zy-xz*zz
        C_gl[:,1,4] = xx*zy+zx*xy
        C_gl[:,2,0] = rt3*yz*zz
        C_gl[:,2,1] = 2*yx*zz+xy
        C_gl[:,2,2] = -xx+2*yy*zz
        C_gl[:,2,3] = -2*yy*zy-yz*zz
        C_gl[:,2,4] = yx*zy+zx*yy
        C_gl[:,3,0] = rt3*(-2*yz**2-zz**2+1)/2
        C_gl[:,3,1] = -2*yx*yz-zx*zz
        C_gl[:,3,2] = -2*yy*yz-zy*zz
        C_gl[:,3,3] = (4*yy**2+2*zy**2+2*yz**2+zz**2-3)/2
        C_gl[:,3,4] = -2*yx*yy-zx*zy
        C_gl[:,4,0] = rt3*xz*yz
        C_gl[:,4,1] = xx*yz+yx*xz
        C_gl[:,4,2] = xy*yz+yy*xz
        C_gl[:,4,3] = -2*xy*yy-xz*yz
        C_gl[:,4,4] = xx*yy+yx*xy
        # rotate
        Qrot[:,4:9] = np.einsum('ijk,ik->ij', C_gl, quadrupoles)
     
    return Qrot

def rot_local2global(Q, R, lmax):
    # for local to global rotation, simply rotate according to R.T
    return rot_global2local(Q, np.swapaxes(R, 1, 2), lmax)

def normalize(matrix, axis=1, ord=2):
    '''
    Normalize a matrix along one dimension
    '''
    matrix /= np.linalg.norm(matrix, axis=axis, keepdims=True, ord=ord)
    return matrix

