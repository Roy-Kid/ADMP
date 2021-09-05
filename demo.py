from python.utils import convert_cart2harm, rot_local2global
import numpy as np

from python.neighborList import construct_nblist
from python.parser import assemble_covalent, init_residues, read_pdb, read_xml
from python.pme import pme_real, generate_construct_local_frame

pdb = 'tests/samples/waterdimer_aligned.pdb'
xml = 'tests/samples/mpidwater.xml'

# return a dict with raw data from pdb
pdbinfo = read_pdb(pdb)

serials = pdbinfo['serials']
names = pdbinfo['names']
resNames = pdbinfo['resNames']
resSeqs = pdbinfo['resSeqs']
positions = pdbinfo['positions']
charges = pdbinfo['charges']
box = pdbinfo['box'] # a, b, c, α, β, γ
lx = box[0]
ly = box[1]
lz = box[2]

mScales = np.array([0.0, 0.0, 0.0, 1.0])
pScales = np.array([0.0, 0.0, 0.0, 1.0])
dScales = np.array([0.0, 0.0, 0.0, 1.0])

box = np.eye(3)*np.array([lx, ly, lz])

rc = 8 # in Angstrom
ethresh = 1e-4

natoms = len(serials)

# Here are the templates[dict] from xml. 
# atomTemplates are per-atom info,
# residueTemplates contain what atoms in the residue and how they connect. 
atomTemplate, residueTemplate = read_xml(xml)

# then we use the template to init atom and residue objects
# TODO: an atomManager and residueManager
atomDicts, residueDicts = init_residues(serials, names, resNames, resSeqs, positions, charges, atomTemplate, residueTemplate)

Q = np.vstack(
    [(atom.c0, atom.dX*10, atom.dY*10, atom.dZ*10, atom.qXX*300, atom.qYY*300, atom.qZZ*300, atom.qXY*300, atom.qXZ*300, atom.qYZ*300) for atom in atomDicts.values()]
)

Qlocal = convert_cart2harm(Q, 2)

axis_type = np.array(
    [atom.axisType for atom in atomDicts.values()]
)

axis_indices = np.vstack(
    [atom.axis_indices for atom in atomDicts.values()]
)

covalent_map = assemble_covalent(residueDicts, natoms)    


## TODO: setup kappa

kappa = 0.328532611

import scipy
from scipy.stats import special_ortho_group

scipy.random.seed(1000)
R1 = special_ortho_group.rvs(3)
R2 = special_ortho_group.rvs(3)

positions[0:3] = positions[0:3].dot(R1)
positions[3:6] = positions[3:6].dot(R2)
positions[3:] += np.array([3.0, 0.0, 0.0])

neighList = construct_nblist(positions, box, rc)

local_frames = generate_construct_local_frame(axis_type, axis_indices)(positions, box)

Qglobal = rot_local2global(Qlocal, local_frames, 2)

e = pme_real(positions, box, Qglobal, kappa, covalent_map, mScales, pScales, dScales, neighList)

print(e)

from openmm.app import *
from openmm import *
from openmm.unit import *
import mpidplugin

pdb = PDBFile(pdb)
forcefield = ForceField(xml)
system = forcefield.createSystem(pdb.topology, nonbondedMethod=LJPME, nonbondedCutoff=8*angstrom, constraints=HBonds, defaultTholeWidth=8)
integrator = VerletIntegrator(1e-10*femtoseconds)
simulation = Simulation(pdb.topology, system, integrator)
context = simulation.context
context.setPositions(positions*angstrom)
state = context.getState(getEnergy=True, getPositions=True)
Etot = state.getPotentialEnergy().value_in_unit(kilojoules_per_mole)
