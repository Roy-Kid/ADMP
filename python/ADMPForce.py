# date: 2021-08-24
# version: 0.0.1

from jax.test_util import check_grads
import numpy as np
import re
from scipy.sparse import csr_matrix
import simtk.openmm.app.element as elem
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from .neighborList import construct_nblist
from .utils import *
from .pme import pme_real, pme_reciprocal, pme_self, pme_reciprocal_force
import mpidplugin

# see the python/mpidplugin.i code 
ZThenX = 0
Bisector = 1
ZBisect = 2
ThreeFold = 3
Zonly = 4
NoAxisType = 5
LastAxisTypeIndex = 6

# ----- pre-process section ----- #
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

def get_mtpls_global(positions, box, params, lmax=2):
    '''
    This function constructs the local frames for each site, and also convert local cartesian into global harmonics
    Essentially, a python version of applyRotationMatrixToParticle in MPIDReferenceForce.cpp

    Inputs:
        positions:
            N * 3: the positions matrix
        params:
            dictionary, coming from read_mpid_inputs
        lmax:
            int, largest l

    Outputs:
        Q: 
            N*(lmax+1)^2, the multipole moments in global harmonics.
        local_frames:
            N*3*3, the local frames, axes arranged in rows
    '''
    # construct local frame
    # see applyRotationMatrixToParticle
    if lmax > 2:
        sys.exit('l > 2 (beyond quadrupole) not supported')
    n_sites = positions.shape[0]
    n_harms = (lmax + 1)**2
    box_inv = np.linalg.inv(box)
    # the R matrix
    local_frames = np.zeros((n_sites, 3, 3))

    z_atoms = params['axis_indices'][:, 0]
    x_atoms = params['axis_indices'][:, 1]
    y_atoms = params['axis_indices'][:, 2]
    vec_z = positions[z_atoms] - positions
    vec_z = pbc_shift(vec_z, box, box_inv)
    vec_z = normalize(vec_z, axis=1)

    vec_x = np.zeros((n_sites, 3))
    vec_y = np.zeros((n_sites, 3))

    # Z-Only
    indices_filter = (params['axis_types'] == Zonly)
    if np.sum(indices_filter) > 0:
        filter1 = (abs(vec_z[:,0]) < 0.866)
        filter2 = np.logical_not(filter1)
        filter1 = np.logical_and(indices_filter, filter1)
        filter2 = np.logical_and(indices_filter, filter2)
        vec_x[filter1] = np.array([1.0, 0.0, 0.0])
        vec_x[filter2] = np.array([0.0, 1.0, 0.0])
    # for those that are not Z-Only, get normalized vecX
    indices_filter = np.logical_not(indices_filter)
    vec_x[indices_filter] = positions[x_atoms[indices_filter]] - positions[indices_filter]
    vec_x[indices_filter] = pbc_shift(vec_x[indices_filter], box, box_inv)
    vec_x[indices_filter] = normalize(vec_x[indices_filter], axis=1)
    # Bisector
    indices_filter = (params['axis_types'] == Bisector)
    if np.sum(indices_filter) > 0:
        vec_z[indices_filter] += vec_x[indices_filter]
        vec_z[indices_filter] = normalize(vec_z[indices_filter], axis=1)
    # z-bisector
    indices_filter = (params['axis_types'] == ZBisect)
    if np.sum(indices_filter) > 0:
        vec_y[indices_filter] = positions[y_atoms[indices_filter]] - positions[indices_filter]
        vec_y[indices_filter] = pbc_shift(vec_y[indices_filter], box, box_inv)
        vec_y[indices_filter] = normalize(vec_y[indices_filter], axis=1)
        vec_x[indices_filter] += vec_y[indices_filter]
        vec_x[indices_filter] = normalize(vec_x[indices_filter], axis=1)
    # ThreeFold
    indices_filter = (params['axis_types'] == ThreeFold)
    if np.sum(indices_filter) > 0:
        vec_y[indices_filter] = positions[y_atoms[indices_filter]] - positions[indices_filter]
        vec_y[indices_filter] = pbc_shift(vec_y[indices_filter], box, box_inv)
        vec_y[indices_filter] = normalize(vec_y[indices_filter], axis=1)
        vec_z[indices_filter] += (vec_x[indices_filter] + vec_y[indices_filter])
        vec_z[indices_filter] = normalize(vec_z[indices_filter])

    # up to this point, z-axis should already be set up and normalized
    tmp = np.einsum('ij,ij->i', vec_x, vec_z)
    tmp = np.expand_dims(tmp, axis=1)
    vec_x -= vec_z * tmp
    vec_x = normalize(vec_x, axis=1)
    # up to this point, x-axis should be ready
    vec_y = np.cross(vec_z, vec_x)

    local_frames = np.stack((vec_x, vec_y, vec_z), axis=1)

    multipoles_lc = np.concatenate((np.expand_dims(params['charges'], axis=1), params['dipoles'], params['quadrupoles']), axis=1)
    multipoles_lh = convert_cart2harm(multipoles_lc, lmax=2)
    Q = rot_local2global(multipoles_lh, local_frames, lmax=2)

    return Q, local_frames

def read_mpid_inputs(pdb, xml):
    '''
    This function read the MPID inputs, including a pdb file and a force definition xml file

    Inputs:
        pdb:
            String, the pdb file name
        xml:
            String, the xml file name

    Outputs:
        positions:
            nx3 float array, the position matrix defined by the pdb file
        list_elems:
            length n, list of OpenMM element objects
        params:
            a dictionary with complete information of charges, dip, quad, oct, pol, thole, axis types and indices etc.
            one should be able to use this information to do multipole PME
            the multipoles are returned using the convention defined in Anthony's book
            ----
            Note params[covalent_map] is stored in csr sparse matrix, with a dimension of NxN, if two atoms are covalent
            neighbors, map[i,j] = n, with n being the number of bonds between them.

    '''
    # first, use the MPID input mechanism to fetch topology and multipole information
    pdb = PDBFile(pdb)
    forcefield = ForceField(xml)
    system = forcefield.createSystem(pdb.topology, nonbondedMethod=LJPME, defaultTholeWidth=5)

    generators = {}
    # print([i.__repr__() for i in forcefield.getGenerators()])
    for force_gen in forcefield.getGenerators():
        generator_name = re.search('.([A-Za-z]+) +object', force_gen.__repr__()).group(1)
        generators[generator_name] = force_gen
    # the mpid force object
    force = generators['MPIDGenerator'].force
    n_sites = force.getNumMultipoles()

    params = {}
    # read in multipoles in local cartesian frame
    charges = []
    dipoles = []
    quadrupoles = []
    octupoles = []
    axis_types = []
    axis_indices = [] # Z, X, Y, as specified in python/mpidplugin.i
    tholes = []
    polarizabilities = []
    # NOTE: pay attention to the ordering of high order cartesian multipoles
    xx, xy, yy, xz, yz, zz = list(range(0, 6))
    xxx, xxy, xyy, yyy, xxz, xyz, yyz, xzz, yzz, zzz = list(range(10))
    order_quad = np.array([xx, yy, zz, xy, xz, yz], dtype=int)
    order_octu = np.array([xxx, xxy, xyy, yyy, xxz, xyz, yyz, xzz, yzz, zzz], dtype=int)
    for i in range(n_sites):
        p = force.getMultipoleParameters(i)
        charges.append(p[0])
        dipoles.append(p[1])
        quadrupoles.append(np.array(p[2])[order_quad])
        octupoles.append(np.array(p[3])[order_octu])
        axis_types.append(p[4])
        axis_indices.append(p[5:8])
        tholes.append(p[8])
        polarizabilities.append(p[9])
    # NOTE: OpenMM defines cartesian moments differently with Anthony's book
    # note the scaling coefficients on quad and oct
    # see loadParticleData function in MPIDReferenceForce.cpp
    params['charges'] = np.array(charges)
    params['dipoles'] = np.array(dipoles) * 10 # now in Angstrom
    params['quadrupoles'] = np.array(quadrupoles) * 3 * 100
    params['octupoles'] = np.array(octupoles) * 15 * 1000
    params['axis_types'] = np.array(axis_types)
    params['axis_indices'] = np.array(axis_indices)
    params['tholes'] = np.array(tholes)
    params['polarizabilities'] = np.array(polarizabilities)

    # topological information, related to real space exclusions
    covalent_map = np.zeros((n_sites, n_sites))
    for i in range(n_sites):
        for n in range(5): # recording up to range 1-6 interactions
            for j in force.getCovalentMap(i, n):
                covalent_map[i, j] = n + 1
    params['covalent_map'] = csr_matrix(covalent_map, dtype=int)

    positions = np.array(pdb.positions.value_in_unit(angstrom))
    box = np.array(pdb.topology.getPeriodicBoxVectors().value_in_unit(angstrom))
    list_elems = []
    for a in pdb.topology.atoms():
        list_elems.append(a.element)

    return positions, box, list_elems, params

# --- end pre-process section --- #

class ADMPGenerator:

    def __init__(self, pdb, xml, rc, ethresh, mScales, pScales, dScales, **kwargs) -> None:
        self.positions, self.box, self.list_elem, self.mpid_params = read_mpid_inputs(pdb, xml)
        
        self.rc = rc
        self.ethresh = ethresh
        self.mScales = mScales
        self.pScales = pScales
        self.dScales = dScales
        

    def create_force(self):
        
        force = ADMPForce()
        force.positions = self.positions
        force.box = self.box
        force.box_inv = np.linalg.inv(self.box)
        force.rc = self.rc
        force.mpid_params = self.mpid_params
        force.ethresh = self.ethresh
        force.mScales = self.mScales
        force.pScales = self.pScales
        force.dScales = self.dScales
        force.box = self.box
        force.lmax = 2
        return force


class ADMPBaseForce:
    
    def __init__(self) -> None:
        pass
    
    def contruct_neighborlist(self):
        self.neighlist = construct_nblist(self.positions, self.box, self.rc)

    def setup_ewald_parameters(self):
        self.kappa, self.K1, self.K2, self.K3 = setup_ewald_parameters(self.rc, self.ethresh, self.box)
        
    def get_mtpls_global(self):
        self.Q, self.local_frames = get_mtpls_global(box = self.box, positions = self.positions, params=self.mpid_params, lmax=self.lmax)


    def update(self, positions=None, box=None, ):
        """
            This method is used to update parameters, such as positions, box etc.
            And calculate related parameters.
        """
        positions = positions or self.positions
        box = box or self.box
        rc = self.rc
        
        self.contruct_neighborlist()
        
        self.setup_ewald_parameters()
        
        self.get_mtpls_global()


class ADMPForce(ADMPBaseForce):
    
    def __init__(self) -> None:
        super().__init__()
    
    def calc_real_space_energy(self):
        return pme_real(self.positions, self.box, self.Q, self.lmax, self.box_inv, self.mpid_params['polarizabilities'], self.rc, self.kappa, self.mpid_params['covalent_map'], self.mScales, self.pScales, self.dScales, self.neighlist)
    
    def calc_self_energy(self):
        return pme_self(self.Q, self.lmax, self.kappa)
    
    def calc_reci_space_energy(self):
        N = np.array([self.K1, self.K2, self.K3])
        Q = self.Q[:, :(self.lmax+1)**2].reshape(self.Q.shape[0], (self.lmax+1)**2)
        return pme_reciprocal(self.positions, self.box, Q, self.lmax, self.kappa, N)
    
    def calc_reci_space_force(self):
        N = np.array([self.K1, self.K2, self.K3])
        Q = self.Q[:, :(self.lmax+1)**2].reshape(self.Q.shape[0], (self.lmax+1)**2)
        check_grads(pme_reciprocal, (self.positions, self.box, Q, self.lmax, self.kappa, N), order = 1)
        return pme_reciprocal_force(self.positions, self.box, Q, self.lmax, self.kappa, N)