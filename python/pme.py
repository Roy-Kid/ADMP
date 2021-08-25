from math import erf

import numpy as np

from .utils import rot_global2local

dielectric = 1389.35455846 # in e^2/A

def calc_ePermCoef(mscales, kappa, dr):

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
    
    rInvVec = np.array([prefactor/dr**i for i in range(0, 9)])
    assert (rInvVec[1] == prefactor / dr).all()
    
    alphaRVec = np.array([(kappa*dr)**i for i in range(0, 10)])
    assert (alphaRVec[1] == kappa * dr).all()
    
    X = 2 * np.exp( -alphaRVec[2] ) / np.sqrt(np.pi)
    tmp = np.array(alphaRVec[1])
    doubleFactorial = 1
    facCount = 1
    erfAlphaR = np.array(list(map(erf, alphaRVec[1])))
    bVec = np.empty((6, len(erfAlphaR)))
    bVec[1] = -erfAlphaR
    for i in range(2, 6):
        bVec[i] = bVec[i-1]+tmp*X/doubleFactorial
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
    dq_m1 = -np.sqrt(3)*rInvVec[4]*(mscales+bVec[3])
    
    ## Q-Q
    
    qq_m0 = rInvVec[5] * (6* (mscales + bVec[4])+ (4/45)* (-3 + 10*alphaRVec[2]) * alphaRVec[5]*X)
    qq_m1 = -(4/15)*rInvVec[5]*(15*(mscales+bVec[4]) + alphaRVec[5]*X)
    qq_m2 = rInvVec[5]*(mscales + bVec[4] - (4/15)*alphaRVec[5]*X)

    return cc, cd, dd_m0, dd_m1, cq, dq_m0, dq_m1, qq_m0, qq_m1, qq_m2


def pme_real(positions, box, Q, lmax, mu_ind, polarizabilities, rc, kappa, covalent_map, mScales, pScales, dScales, nbs_obj):
    '''
    This function computes the realspace part of PME interactions.

    Inputs:
        positions:
            N * 3, positions of all atoms, in ANGSTROM
        box:
            3 * 3, box size, axes arranged in rows, in ANGSTROM
        Q:
            N * (lmax+1)^2, fixed multipole moments in global spherical harmonics
        lmax:
            int, maximum order of multipoles
        mu_ind:
            N * 3, induced dipoles, in global spherical harmonics
        polarizabilities:
            N * 3 * 3, the polarizability tensor of each site, UNIT TO BE DETERMINED...
        rc:
            float, cutoff distance, in ANGSTROM
        kappa:
            float, the short range attenuation in PME, in ANGSTROM^-1
        covalent_map:
            N * N, in scipy csr sparse matix form, each row specifies the covalent neighbors of each atom
            0 means not covalent neighor, 1 means one bond away, 2 means two bonds away etc.
        mScales:
            len(n) vector, the fixed multipole - fixed multipole interaction damping coefficients, (n) is the
            maximum exclusion distance
        pScales:
            len(n) vector, the fixed multipole - induced dipole scalings
        dScales:
            len(n) vector, the induced - induced dipole scalings
        obs_obj:
            neighbor list objects, as in the ml_ff module

    Outputs:
        ene_real: 
            float, real space energy
    '''
    
    # The following is written by Roy Kid
    # any confusion please email lijichen365@126.com directly 

    # a bit preprocessing to scaling factors
    max_excl = len(mScales)
    assert(len(mScales) == len(pScales))
    assert(len(mScales) == len(dScales))
    
    # an explain: 
    # "1" padding is for <get [mpd]scale> section
    mScales = np.concatenate([mScales, np.array([1])])
    pScales = np.concatenate([pScales, np.array([1])])
    dScales = np.concatenate([dScales, np.array([1])])


    covalent_map = covalent_map.multiply(covalent_map<=max_excl)

    # neighbor list
    n_nbs = nbs_obj.n_nbs
    max_n_nbs = np.max(n_nbs)
    nbs = nbs_obj.nbs[:, 0:max_n_nbs]
    dr_vecs = nbs_obj.dr_vecs[:, 0:max_n_nbs]
    drdr = nbs_obj.distances2[:, 0:max_n_nbs]
    box_inv = np.linalg.inv(box)
    n_atoms = len(positions)
    
    t = np.sum(n_nbs)
    tot_pair = np.empty((t, 2), dtype=int)
    drs = np.empty((t, 3))
    drdrs = np.empty((t, ))
    for i_atom in range(n_atoms):
        
        start = np.sum(n_nbs[0:i_atom])
        end = np.sum(n_nbs[0:i_atom]) + n_nbs[i_atom]
        
        tot_pair[start: end, 0] = i_atom
        tot_pair[start: end, 1] = nbs[i_atom, 0: n_nbs[i_atom]]
        
        drs[start: end, :] = dr_vecs[i_atom, 0: n_nbs[i_atom]]
        drdrs[start: end] = drdr[i_atom, 0: n_nbs[i_atom]]
        
        
    # a filter that exclude those pairs 
    # that atomid of pair[0] < pair[1]
    # to avoid double counting
    pairs = tot_pair[tot_pair[:, 0] < tot_pair[:, 1]]
    drs = drs[tot_pair[:, 0] < tot_pair[:, 1]]
    drdrs = drdrs[tot_pair[:, 0] < tot_pair[:, 1]]
    distances = drdrs**0.5
    assert len(pairs) == len(drs)
    npair = len(pairs)

    mscales = np.empty((npair, ))
    Ri = np.empty((npair, 3, 3))
    Q_extendi = np.empty((npair, Q.shape[1]))
    Q_extendj = np.empty((npair, Q.shape[1]))
    
    for i, pair in enumerate(pairs):
        
        # get [m-p-d]scale
        cmap = covalent_map[pair[0], :].toarray()
        nbond = cmap[0, pair[1]]
        mscale = mScales[nbond-1]
        mscales[i] = mscale

        # form rotation matrix
        # transfer from MPIDReferenceForce.cpp line 1055
        iPos = positions[pair[0]]
        jPos = positions[pair[1]]
        deltaR = drs[i]
        r = distances[i]
        vectorZ = deltaR/r
        vectorX = np.array(vectorZ)
        if iPos[1] != jPos[1] or iPos[2] != jPos[2]:
            vectorX[0] += 1
        else:
            vectorX[1] += 1
        dot = np.dot(vectorZ, vectorX)
        vectorX -= vectorZ*dot
        vectorX = vectorX / np.linalg.norm(vectorX)
        vectorY = np.cross(vectorZ, vectorX)
        Ri[i] = np.array([vectorX, vectorY, vectorZ])

        # match Q global with pair
        # pair:= [[i, j]] 
        # Qi  :=  [i]       
        # Qj  :=     [j]
        Q_extendi[i] = Q[pair[0]]
        Q_extendj[i] = Q[pair[1]]

    # as we extending to octupole
    # todo: 2->3
    # add lmax 2->3 support in rot_global2local
    qiQI = rot_global2local(Q_extendi, Ri, 2)
    qiQJ = rot_global2local(Q_extendj, Ri, 2)

    # TODO: support lmax to control what order needed
    cc, cd, dd_m0, dd_m1, cq, dq_m0, dq_m1, qq_m0, qq_m1, qq_m2 = calc_ePermCoef(mscales, kappa, distances)
    
    # as we extending to octupole
    # todo: 9->16
    Vij = np.empty((len(qiQJ), 9))
    Vji = np.empty((len(qiQI), 9))
    
    # transfer from
    # calculatePmeDirectElectrostaticPairIxn
    # C-C

    # The vertorize is used
    # once extend to octupole
    # remember that add a column 
    # in the front of index of Vij/qiQJ etc.
    # all the pair's info are hstack to form 2D array.
    # MPID: Vij[0] = ePermCoeff*qiQJ[0]
    # pme : Vij[:, 0] = cc*qiQJ[:, 0]
    # to calculate all the pairs at one time

    Vij[:, 0] = cc*qiQJ[:, 0]
    Vji[:, 0] = cc*qiQI[:, 0]

    # C-D 

    Vij[:, 0] += -cd*qiQJ[:, 1]
    Vji[:, 1] = -cd*qiQI[:, 0]
    Vij[:, 1] = cd*qiQJ[:, 0]
    Vji[:, 0] += cd*qiQI[:, 1]

    # D-D m0 

    Vij[:, 1] += dd_m0 * qiQJ[:, 1]
    Vji[:, 1] += dd_m0 * qiQI[:, 1]
    
    # D-D m1 

    Vij[:, 2] = dd_m1*qiQJ[:, 2]
    Vji[:, 2] = dd_m1*qiQI[:, 2]
    Vij[:, 3] = dd_m1*qiQJ[:, 3]
    Vji[:, 3] = dd_m1*qiQI[:, 3]

    # C-Q
    
    Vij[:, 0]  += cq*qiQJ[:, 4]
    Vji[:, 4]   = cq*qiQI[:, 0]
    Vij[:, 4]   = cq*qiQJ[:, 0]
    Vji[:, 0]  += cq*qiQI[:, 4]
    
    # D-Q m0
    
    Vij[:, 1]  += dq_m0*qiQJ[:, 4]
    Vji[:, 4]  += dq_m0*qiQI[:, 1] 
    
    # Q-D m0
    
    Vij[:, 4]  += -(dq_m0*qiQJ[:, 1])
    Vji[:, 1]  += -(dq_m0*qiQI[:, 4])
    
    # D-Q m1
    
    Vij[:, 2]  += dq_m1*qiQJ[:, 5]
    Vji[:, 5]   = dq_m1*qiQI[:, 2]

    Vij[:, 3]  += dq_m1*qiQJ[:, 6]
    Vji[:, 6]   = dq_m1*qiQI[:, 3]

    Vij[:, 5]   = -(dq_m1*qiQJ[:, 2])
    Vji[:, 2]  += -(dq_m1*qiQI[:, 5])
    
    Vij[:, 6]   = -(dq_m1*qiQJ[:, 3])
    Vji[:, 3]  += -(dq_m1*qiQI[:, 6])
    
    # Q-Q m0
    
    Vij[:, 4]  += qq_m0*qiQJ[:, 4]
    Vji[:, 4]  += qq_m0*qiQI[:, 4]    
    
    # Q-Q m1
    
    Vij[:, 5]  += qq_m1*qiQJ[:, 5]
    Vji[:, 5]  += qq_m1*qiQI[:, 5]
    Vij[:, 6]  += qq_m1*qiQJ[:, 6]
    Vji[:, 6]  += qq_m1*qiQI[:, 6]
    
    # Q-Q m2
    
    Vij[:, 7]  = qq_m2*qiQJ[:, 7]
    Vji[:, 7]  = qq_m2*qiQI[:, 7]
    Vij[:, 8]  = qq_m2*qiQJ[:, 8]
    Vji[:, 8]  = qq_m2*qiQI[:, 8]
    
    return 0.5*(np.sum(qiQI*Vij)+np.sum(qiQJ*Vji))

def pme_self(Q, lmax, kappa):
    '''
    This function calculates the PME self energy

    Inputs:
        Q:
            N * (lmax+1)^2: the global harmonics multipoles
        lmax:
            int: largest L
        kappa:
            float: kappa used in PME

    Output:
        ene_self:
            float: the self energy
    '''
    n_atoms = Q.shape[0]
    n_harms = (lmax + 1) ** 2
    Q = Q[:, 0:n_harms]  # only keep the terms we care
    l_list = np.array([0], dtype=np.int64)
    l_fac2 = np.array([1], dtype=np.int64)
    tmp = 1
    for l in range(1, lmax+1):
        l_list = np.concatenate((l_list, np.ones(2*l+1, dtype=np.int64)*l))
        tmp *= (2*l + 1)
        l_fac2 = np.concatenate((l_fac2, np.ones(2*l+1, dtype=np.int64)*tmp))

    l_list = np.repeat(l_list, n_atoms).reshape((-1, n_atoms)).T
    l_fac2 = np.repeat(l_fac2, n_atoms).reshape((-1, n_atoms)).T
    factor = kappa/np.sqrt(np.pi) * (2*kappa*kappa)**l_list / l_fac2
    return - np.sum(factor * Q**2) * dielectric

def pme_reciprocal(positions, box, Q, lmax, kappa, N):
    
    padder = np.arange(-3, 3)
    shifts = np.array(np.meshgrid(padder, padder, padder)).T.reshape((1, 216, 3))
    
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
        Nj_Aji_star = (N.reshape((1, 3)) * np.linalg.inv(box)).T
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
        
        R_in_m_basis =  np.einsum("ij,kj->ki", Nj_Aji_star, R_a)
        
        m_u0 = np.empty_like(R_in_m_basis, dtype=np.int32)
        np.ceil(R_in_m_basis, m_u0, casting='unsafe')
        
        u0 = (m_u0 - R_in_m_basis) + 6/2
        return m_u0, u0

    def bspline6(u):
        """
        Computes the cardinal B-spline function
        """
        return np.piecewise(u, 
                            [np.logical_and(u>=0, u<1.), 
                            np.logical_and(u>=1, u<2.), 
                            np.logical_and(u>=2, u<3.), 
                            np.logical_and(u>=3, u<4.), 
                            np.logical_and(u>=4, u<5.), 
                            np.logical_and(u>=5, u<6.)],
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
        return np.piecewise(u, 
                            [np.logical_and(u>=0., u<1.), 
                            np.logical_and(u>=1., u<2.), 
                            np.logical_and(u>=2., u<3.), 
                            np.logical_and(u>=3., u<4.), 
                            np.logical_and(u>=4., u<5.), 
                            np.logical_and(u>=5., u<6.)],
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
        return np.piecewise(u, 
                            [np.logical_and(u>=0., u<1.), 
                            np.logical_and(u>=1., u<2.), 
                            np.logical_and(u>=2., u<3.), 
                            np.logical_and(u>=3., u<4.), 
                            np.logical_and(u>=4., u<5.), 
                            np.logical_and(u>=5., u<6.)],
                            [lambda u: u**3/6,
                            lambda u: u**3/6 - (u - 1)**3,
                            lambda u: 5*u**3/3 - 12*u**2 + 27*u - 19,
                            lambda u: -5*u**3/3 + 18*u**2 - 63*u + 71,
                            lambda u: 5*u**3/6 - 12*u**2 + 57*u - 89,
                            lambda u: -u**3/6 + 3*u**2 - 18*u + 36,] )


    def theta_eval(u):
        """
        Evaluates the value of theta given 3D u values at ... points 
        
        Input:
            u:
                ... x 3 matrix

        Output:
            theta:
                ... matrix
        """
        theta = np.prod(bspline6(u), axis = -1)
        return theta

    def thetaprime_eval(u, Nj_Aji_star, theta):
        """
        First derivative of theta with respect to x,y,z directions
        
        Input:
            u
            Nj_Aji_star:
                reciprocal lattice vectors
            theta
        
        Output:
            N_a * 3 matrix
        """

        M_u = bspline6(u)
        M_u[M_u == 0] = np.inf
        
        # Notice that u = m_u0 - R_in_m_basis + 6/2
        # therefore the Jacobian du_j/dx_i = - Nj_Aji_star
        return np.einsum("ij,kj->ki", -Nj_Aji_star, theta.reshape((-1, 1)) * bspline6prime(u)/M_u)

    def theta2prime_eval(u, Nj_Aji_star):
        """
        compute the 3 x 3 second derivatives of theta with respect to xyz
        
        Input:
            u
            Nj_Aji_star
        
        Output:
            N_A * 3 * 3
        """
        M_u = bspline6(u)
        Mprime_u = bspline6prime(u)
        M2prime_u = bspline6prime2(u)

        div = np.empty((u.shape[0], 3, 3))

        div[:, 0, 0] = M2prime_u[:, 0] * M_u[:, 1] * M_u[:, 2]
        div[:, 1, 1] = M2prime_u[:, 1] * M_u[:, 0] * M_u[:, 2]
        div[:, 2, 2] = M2prime_u[:, 2] * M_u[:, 0] * M_u[:, 1]

        div[:, 0, 1] = Mprime_u[:, 0] * Mprime_u[:, 1] * M_u[:, 2]
        div[:, 0, 2] = Mprime_u[:, 0] * Mprime_u[:, 2] * M_u[:, 1]
        div[:, 1, 2] = Mprime_u[:, 1] * Mprime_u[:, 2] * M_u[:, 0]
        div[:, 1, 0] = div[:, 0, 1]
        div[:, 2, 0] = div[:, 0, 2]
        div[:, 2, 1] = div[:, 1, 2]
        
        # Notice that u = m_u0 - R_in_m_basis + 6/2
        # therefore the Jacobian du_j/dx_i = - Nj_Aji_star
        return np.einsum("im,jn,kmn->kij", -Nj_Aji_star, -Nj_Aji_star, div)

    def sph_harmonics_GO(u0, Nj_Aji_star, lmax):
        '''
        Find out the value of spherical harmonics GRADIENT OPERATORS, assume the order is:
        00, 10, 11c, 11s, 20, 21c, 21s, 22c, 22s, ...
        Currently supports lmax <= 2

        Inputs:
            u0: 
                a N_a * 3 matrix containing all positions
            Nj_Aji_star:
                reciprocal lattice vectors in the m-grid
            lmax: 
                the maximum l value

        Output: 
            harmonics: 
                a Na * (6**3) * (l+1)^2 matrix, STGO operated on theta,
                evaluated at 6*6*6 integer points about reference points m_u0 
        '''
        if lmax > 2:
            raise NotImplementedError('l > 2 (beyond quadrupole) not supported')
        n_harm = (lmax + 1)**2

        N_a = u0.shape[0]
        u = (u0[:, np.newaxis, :] + shifts).reshape((N_a*216, 3)) 

        harmonics = np.zeros((N_a*216, n_harm))
        
        theta = theta_eval(u)
        harmonics[:, 0] = theta
        
        if lmax >= 1:
            # dipole
            thetaprime = thetaprime_eval(u, Nj_Aji_star, theta)
            harmonics[:, 1] = thetaprime[:, 2]
            harmonics[:, 2] = thetaprime[:, 0]
            harmonics[:, 3] = thetaprime[:, 1]
        if lmax >= 2:
            # quadrapole
            theta2prime = theta2prime_eval(u, Nj_Aji_star)
            rt3 = np.sqrt(3)
            harmonics[:, 4] = (3*theta2prime[:,2,2] - np.trace(theta2prime, axis1=1, axis2=2)) / 2
            harmonics[:, 5] = rt3 * theta2prime[:, 0, 2]
            harmonics[:, 6] = rt3 * theta2prime[:, 1, 2]
            harmonics[:, 7] = rt3/2 * (theta2prime[:, 0, 0] - theta2prime[:, 1, 1])
            harmonics[:, 8] = rt3 * theta2prime[:, 0, 1]
        
        harmonics = harmonics.reshape(N_a, 216, n_harm)

        return harmonics.reshape(N_a, 216, n_harm)

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
        
        Q_dbf = Q
        
        if lmax >= 1:
            Q_dbf[:,1:4] = Q[:,1:4] 
        if lmax >= 2:
            Q_dbf[:,4:9] = Q[:,4:9] / 3
        
        Q_m_pera = np.sum( Q_dbf[:,np.newaxis,:]* sph_harms, axis=2)

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
        Q_mesh = np.zeros((N[0], N[1], N[2]))
        for ai in range(m_u0.shape[0]):
            for i in range(shifts[0].shape[0]):
                shift = shifts[0, i]
                shift = shift.reshape(3)
                indices = np.mod(m_u0[ai] + shift, N)
                Q_mesh[indices[0], indices[1], indices[2]] += Q_mesh_pera[ai, i]

        return Q_mesh

    def setup_kpts_integer(N):
        """
        Outputs:
            kpts_int:
                n_k * 3 matrix, n_k = N[0] * N[1] * N[2]
        """
        N_half = N.reshape(3)

        kx, ky, kz = [np.roll(np.arange(- (N_half[i] - 1) // 2, (N_half[i] + 1) // 2 ), - (N_half[i] - 1) // 2) for i in range(3)]

        kpts_int = np.array([ki.flatten() for ki in np.meshgrid(kx, ky, kz)]).T

        kpts_int = kpts_int @ np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])

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
        box_inv = np.linalg.inv(box)

        # K * 3, coordinate in reciprocal space
        kpts = 2 * np.pi * kpts_int.dot(box_inv)

        ksr = np.sum(kpts**2, axis=1)

        # 4 * K
        kpts = np.hstack((kpts, ksr[:, np.newaxis])).T

        return kpts

    def E_recip_on_grid(Q_mesh, box, N, kappa):
        """
        Computes the reciprocal part energy
        """
        
        N = N.reshape(1,1,3)
        kpts_int = setup_kpts_integer(N)
        kpts = setup_kpts(box, kpts_int)
        kpts[3, 0] = np.nan

        m = np.linspace(-3,3,7).reshape(7, 1, 1)
        # theta_k : array of shape n_k
        theta_k = np.prod(
            np.sum(
                bspline6(m + 6/2) * np.cos(2*np.pi*m*kpts_int[np.newaxis] / N),
                axis = 0
            ),
            axis = 1
        )

        S_k = np.fft.fftn(Q_mesh)

        S_k = S_k.flatten()

        E_k = 2*np.pi/kpts[3]/np.linalg.det(box) * np.exp( - kpts[3] /4 /kappa**2) * np.abs(S_k/theta_k)**2
        E_k[0] = 0
        return np.sum(E_k)
    
    Nj_Aji_star = get_recip_vectors(N, box)
    m_u0, u0 = u_reference(positions, Nj_Aji_star)
    sph_harms = sph_harmonics_GO(u0, Nj_Aji_star, lmax)
    Q_mesh_pera = Q_m_peratom(Q, sph_harms)
    Q_mesh = Q_mesh_on_m(Q_mesh_pera, m_u0, N)
    
    # Outputs energy in OPENMM units    
    return E_recip_on_grid(Q_mesh, box, N, kappa)*dielectric