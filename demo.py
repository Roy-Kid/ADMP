from email import generator
from python.utils import convert_cart2harm
import numpy as np
from simtk.openmm.openmm import Platform_getPluginLoadFailures
from python.ADMPForce import ADMPGenerator, get_mtpls_global
import scipy
from scipy.stats import special_ortho_group
from simtk.openmm import *
from simtk.openmm.app import *
from simtk.unit import *
import mpidplugin
import python.pme as pme

mScales = np.array([0.0, 0.0, 0.0, 1.0])
pScales = np.array([0.0, 0.0, 0.0, 1.0])
dScales = np.array([0.0, 0.0, 0.0, 1.0])
rc = 8 # in Angstrom
ethresh = 1e-4


pdb = 'tests/samples/waterdimer_aligned.pdb'
xml = 'tests/samples/mpidwater.xml'
generator = ADMPGenerator(pdb, xml, rc, ethresh, mScales, pScales, dScales, )
# get a random geometry for testing
scipy.random.seed(1000)
R1 = special_ortho_group.rvs(3)
R2 = special_ortho_group.rvs(3)

positions = generator.positions
positions[0:3] = positions[0:3].dot(R1)
positions[3:6] = positions[3:6].dot(R2)
positions[3:] += np.array([3.0, 0.0, 0.0])


force = generator.create_force()
force.update()
force.kappa = 0.328532611

multipoles_lc = np.concatenate((np.expand_dims(force.mpid_params['charges'], axis=1), force.mpid_params['dipoles'], force.mpid_params['quadrupoles']), axis=1)
Q_lh = convert_cart2harm(multipoles_lc, lmax=2)
axis_types = force.mpid_params['axis_types']
axis_indices = force.mpid_params['axis_indices']

# test reci_space_energy
print('============reciprocal energy===========')
reci_e = force.calc_reci_space_energy()
print(reci_e)
print('============reciprocal force============')
reci_f = force.calc_reci_space_force()
print(reci_f)



# test real_space_energy
# real_e = force.calc_real_space_energy()

# # test self energy
# self_e = force.calc_self_energy()


# print('real space', real_e)
# print('self', self_e)

# pdb = PDBFile(pdb)
# forcefield = ForceField(xml)
# system = forcefield.createSystem(pdb.topology, nonbondedMethod=LJPME, nonbondedCutoff=8*angstrom, constraints=HBonds, defaultTholeWidth=8)
# integrator = VerletIntegrator(1e-10*femtoseconds)
# simulation = Simulation(pdb.topology, system, integrator)
# context = simulation.context
# context.setPositions(positions*angstrom)
# state = context.getState(getEnergy=True, getPositions=True)
# Etot = state.getPotentialEnergy().value_in_unit(kilojoules_per_mole)
