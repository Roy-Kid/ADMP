# author: Roy Kid
# contact: lijichen365@126.com
# date: 2021-10-06
# version: 0.0.1

from openmm.app import *
from openmm import *
from openmm.unit import *

nWater = int(input('enter how many H2Os to add: '))
if not isinstance(nWater, int):
    raise TypeError(f'num of H2Os must be int but {type(nWater)}')

print('Loading...')
pdb = PDBFile('./samples/water.pdb')
forcefield = ForceField('amber99sb.xml', 'tip3p.xml')
modeller = Modeller(pdb.topology, pdb.positions)
modeller.deleteWater()

print('Adding water...')
modeller.addSolvent(forcefield, model='tip3p', numAdded=nWater)
print('Minimizing...')
system = forcefield.createSystem(modeller.topology, nonbondedMethod=PME)
integrator = VerletIntegrator(0.001*picoseconds)
simulation = Simulation(modeller.topology, system, integrator)
simulation.context.setPositions(modeller.positions)
simulation.minimizeEnergy(maxIterations=100)
print('Saving...')
positions = simulation.context.getState(getPositions=True).getPositions()
PDBFile.writeFile(simulation.topology, positions, open(f'water{nWater}.pdb', 'w'))
print('Done')