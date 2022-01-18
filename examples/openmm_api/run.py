#!/usr/bin/env python
import sys 
import openmm
import openmm.app as app
import openmm.unit as unit
import numpy as np
import jax.numpy as jnp
from collections import defaultdict
from admp.api import Hamiltonian
from jax_md import space, partition
from jax import grad


if __name__ == '__main__':
    H = Hamiltonian('forcefield.xml')
    app.Topology.loadBondDefinitions("residues.xml")
    pdb = app.PDBFile("water1024.pdb")
    rc = 4.0
    potentials = H.createPotential(pdb.topology, nonbondedCutoff=rc*unit.angstrom)
    # generator stores all force field parameters
    # pot_fn is the actual energy calculator
    generator = H.getGenerators()[0]
    pot_fn = potentials[0]

    # construct inputs
    positions = jnp.array(pdb.positions._value) * 10
    a, b, c = pdb.topology.getPeriodicBoxVectors()
    box = jnp.array([a._value, b._value, c._value]) * 10
    # neighbor list
    displacement_fn, shift_fn = space.periodic_general(box, fractional_coordinates=False)
    neighbor_list_fn = partition.neighbor_list(displacement_fn, box, rc, 0, format=partition.OrderedSparse)
    nbr = neighbor_list_fn.allocate(positions)
    pairs = nbr.idx.T

    print(pot_fn(positions, box, pairs, generator.params))
    param_grad = grad(pot_fn, argnums=3)(positions, box, pairs, generator.params)
    print(param_grad['mScales'])
