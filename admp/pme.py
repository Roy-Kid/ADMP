#!/usr/bin/env python
import numpy as np
from functools import partial
import jax.numpy as jnp
from jax.scipy.special import erf
from admp.settings import jit_condition

# Functions that are related to electrostatic pme


DIELECTRIC = 1389.35455846

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
    kappa = np.sqrt(-np.log(2*ethresh))/rc
    K1 = np.ceil(2 * kappa * box[0, 0] / 3 / ethresh**0.2)
    K2 = np.ceil(2 * kappa * box[1, 1] / 3 / ethresh**0.2)
    K3 = np.ceil(2 * kappa * box[2, 2] / 3 / ethresh**0.2)

    return kappa, int(K1), int(K2), int(K3)


@jit_condition(static_argnums=(3))
def calc_e_perm(dr, mscales, kappa, lmax=2):

    '''
    This function calculates the ePermCoefs at once
    ePermCoefs is basically the interaction tensor between permanent multipole components
    Everything should be done in the so called quasi-internal (qi) frame
    Energy = \sum_ij qiQi * ePermCoeff_ij * qiQj

    Inputs:
        dr: 
            float: distance between one pair of particles
        mscales:
            array: same as pme_realspace()
        kappa:
            float: same as pme_realspace()
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
        
    bVec = jnp.empty((6, len(erfAlphaR)))

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
        cd = rInvVec[2] * ( mscales + bVec[2] )
        # D-D: 2
        dd_m0 = -2/3 * rInvVec[3] * (3*(mscales + bVec[3]) + alphaRVec[3]*X )
        dd_m1 = rInvVec[3] * (mscales + bVec[3] - (2/3)*alphaRVec[3]*X)
    if lmax >= 2:
        ## C-Q: 1
        cq = (mscales + bVec[3])*rInvVec[3]
        ## D-Q: 2
        dq_m0 = rInvVec[4] * (3* (mscales + bVec[3])+ (4/3) * alphaRVec[5]*X)
        dq_m1 = -jnp.sqrt(3)*rInvVec[4]*(mscales+bVec[3])
        ## Q-Q
        qq_m0 = rInvVec[5] * (6* (mscales + bVec[4])+ (4/45)* (-3 + 10*alphaRVec[2]) * alphaRVec[5]*X)
        qq_m1 = -(4/15)*rInvVec[5]*(15*(mscales+bVec[4]) + alphaRVec[5]*X)
        qq_m2 = rInvVec[5]*(mscales + bVec[4] - (4/15)*alphaRVec[5]*X)

    if lmax == 0:
        return cc
    elif lmax == 1:
        return cc, cd, dd_m0, dd_m1
    else:
        return cc, cd, dd_m0, dd_m1, cq, dq_m0, dq_m1, qq_m0, qq_m1, qq_m2


