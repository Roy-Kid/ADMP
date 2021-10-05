
from python.utils import rot_local2global

from python.neighborList import construct_nblist
import numpy as np
from python.ADMPForce import get_mtpls_global, read_mpid_inputs
import scipy
from scipy.stats import special_ortho_group
from simtk.openmm import *
from simtk.openmm.app import *
from simtk.unit import *
from python.pme import pme_real, pme_reciprocal, pme_self
import mpidplugin

mScales = np.array([0.0, 0.0, 0.0, 1.0])
pScales = np.array([0.0, 0.0, 0.0, 1.0])
dScales = np.array([0.0, 0.0, 0.0, 1.0])
rc = 8 # in Angstrom
ethresh = 1e-4
lmax = 2

pdb = 'tests/samples/waterdimer_aligned.pdb'
xml = 'tests/samples/mpidwater.xml'
positions, box, list_elem, mpid_params = read_mpid_inputs(pdb, xml)

scipy.random.seed(1000)
R1 = special_ortho_group.rvs(3)
R2 = special_ortho_group.rvs(3)
polarizabilities = mpid_params['polarizabilities']
covalent_map = mpid_params['covalent_map']

positions[0:3] = positions[0:3].dot(R1)
positions[3:6] = positions[3:6].dot(R2)
positions[3:] += np.array([3.0, 0.0, 0.0])

kappa = 0.328532611
K1 = np.ceil(2 * kappa * box[0, 0] / 3 / ethresh**0.2)
K2 = np.ceil(2 * kappa * box[1, 1] / 3 / ethresh**0.2)
K3 = np.ceil(2 * kappa * box[2, 2] / 3 / ethresh**0.2)


Q_global, Q_local, local_frames = get_mtpls_global(positions, box, mpid_params, lmax)

def calc_energy(positions, box, Q_local, lmax, polarizabilities, rc, kappa, covalent_map, mScales, pScales, dScales):
    box_inv = np.linalg.inv(box)
    Q_global = rot_local2global(Q_local, local_frames, lmax=2)
    neighlist = construct_nblist(positions, box, rc)
    
    energy = 0
    print('positions: ', positions)
    print('box: ', box)
    print('Qlocal: ', Q_local)
    print('local_frame: ', local_frames)
    print('Qglobal: ', Q_global)
    energy += pme_real(positions, box, Q_global, lmax, box_inv, polarizabilities, rc, kappa, covalent_map, mScales, pScales, dScales, neighlist)
    print('real energy: ', energy)
    energy += pme_self(Q_global, lmax, kappa)
    
    N = np.array([int(K1), int(K2), int(K3)])
    Q = Q_global[:, :(lmax+1)**2].reshape(Q_global.shape[0], (lmax+1)**2)
    energy += pme_reciprocal(positions, box, Q, lmax, kappa, N)
    return energy
    
energy = calc_energy(positions, box, Q_local, lmax, polarizabilities, rc, kappa, covalent_map, mScales, pScales, dScales)
print('total_energy: ', energy)


# --- bench mark --- # 
pdb = PDBFile(pdb)
forcefield = ForceField(xml)
system = forcefield.createSystem(pdb.topology, nonbondedMethod=LJPME, nonbondedCutoff=8*angstrom, constraints=HBonds, defaultTholeWidth=8)
integrator = VerletIntegrator(1e-10*femtoseconds)
simulation = Simulation(pdb.topology, system, integrator)
context = simulation.context
context.setPositions(positions*angstrom)
state = context.getState(getEnergy=True, getPositions=True)
Etot = state.getPotentialEnergy().value_in_unit(kilojoules_per_mole)
