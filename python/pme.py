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
    # l_list = np.tile(l_list, (n_atoms, 1))
    l_list = np.repeat(l_list, n_atoms).reshape((-1, n_atoms)).T
    # l_fac2 = np.tile(l_fac2, (n_atoms, 1))
    l_fac2 = np.repeat(l_fac2, n_atoms).reshape((-1, n_atoms)).T
    factor = kappa/np.sqrt(np.pi) * (2*kappa*kappa)**l_list / l_fac2
    return - np.sum(factor * Q**2) * dielectric
