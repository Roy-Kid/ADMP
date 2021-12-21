# import numpy as np
# import sys
# print(sys.path)
# import pytest
# import scipy
# from python.ADMPForce import ADMPGenerator
# from scipy.stats import special_ortho_group
# from simtk.openmm import *
# from simtk.openmm.app import *
# from simtk.unit import *


# @pytest.fixture
# def water_dimer():
#     # basic inputs
#     mScales = np.array([0.0, 0.0, 0.0, 1.0])
#     pScales = np.array([0.0, 0.0, 0.0, 1.0])
#     dScales = np.array([0.0, 0.0, 0.0, 1.0])
#     rc = 8 # in Angstrom
#     ethresh = 1e-4


#     pdb = 'tests/samples/waterdimer_aligned.pdb'
#     xml = 'tests/samples/mpidwater.xml'
#     gen = ADMPGenerator(pdb, xml, rc, ethresh, mScales, pScales, dScales, )
#     # get a random geometry for testing
#     scipy.random.seed(1000)
#     R1 = special_ortho_group.rvs(3)
#     R2 = special_ortho_group.rvs(3)
    
#     positions = gen.positions
#     positions[0:3] = positions[0:3].dot(R1)
#     positions[3:6] = positions[3:6].dot(R2)
#     positions[3:] += np.array([3.0, 0.0, 0.0])
#     yield gen
#     # celllist_1d = neighbor_list.construct_cell_list(positions.T, np.arange(n_atoms, dtype=int), n_atoms, box.T, box_inv.T, rc)
#     # cell_index_map = neighbor_list.build_cell_index_map(celllist_1d, n_atoms)
#     # dimensions = np.flip(celllist_1d[0:4], axis=0)
#     # celllist = celllist_1d[4:].reshape(dimensions)
#     # nbs_obj = neighbor_list.find_neighbors_for_all(positions.T, celllist_1d, n_atoms, rc, box.T, box_inv.T)
#     # nbs, n_nbs, distances2, dr_vecs = neighbor_list.construct_nblist(positions, box, rc)
#     # nbs_obj = neighbor_list.NBLS(nbs, n_nbs, distances2, dr_vecs)
#     # nbs_obj = neighbor_list.construct_nblist(positions, box, rc)

#     # lets not consider induction yet ...
#     # mu_ind = np.zeros((n_atoms, 3))
#     # kappa, K1, K2, K3 = setup_ewald_parameters(rc, ethresh, box)
#     # print(kappa, K1, K2, K3)
#     # kappa = 0.328532611

#     # Q, local_frames = get_mtpls_global(positions, box, mpid_params)

#     # lmax = 2
#     # ene_real = pme_real(positions, box, Q, lmax, mu_ind, mpid_params['polarizabilities'], rc, kappa, mpid_params['covalent_map'], mScales, pScales, dScales, nbs_obj)
#     # ene_self = pme_self(Q, int(lmax), kappa)

#     # print('--------- result ------------')
#     # print('kappa (A^-1):     ', kappa)
#     # print('ene_real: ',   ene_real)
#     # print('ene_self:         ', ene_self)

#     pdb = PDBFile(pdb)
#     forcefield = ForceField(xml)
#     system = forcefield.createSystem(pdb.topology, nonbondedMethod=LJPME, nonbondedCutoff=8*angstrom, constraints=HBonds, defaultTholeWidth=8)
#     integrator = VerletIntegrator(1e-10*femtoseconds)
#     simulation = Simulation(pdb.topology, system, integrator)
#     context = simulation.context
#     context.setPositions(positions*angstrom)
#     state = context.getState(getEnergy=True, getPositions=True)
#     Etot = state.getPotentialEnergy().value_in_unit(kilojoules_per_mole)
