from functools import partial
from time import time
from jax._src.api import value_and_grad

import jax.numpy as jnp
import numpy as np
import pytest
from admp.multipole import rot_local2global
from admp.pme import ADMPPmeForce, pme_real, pme_self, setup_ewald_parameters
from admp.recip import Ck_1, generate_pme_recip
from admp.spatial import generate_construct_local_frames
from jax_md import partition, space
from time import time
from contextlib import contextmanager
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
import numpy.testing as npt

class TestResultWithOldParser:
    
    def test_consistency(self, water, request):

        (
            nwater,
            (
                serials,
                names,
                resNames,
                resSeqs,
                positions,
                box,
                charges,
                Q,
                Q_local,
                axis_types,
                axis_indices,
                covalent_map,
            ),
        ) = water

        mScales = jnp.array([0.0, 0.0, 0.0, 1.0, 1.0])
        pScales = jnp.array([0.0, 0.0, 0.0, 1.0, 1.0])
        dScales = jnp.array([0.0, 0.0, 0.0, 1.0, 1.0])

        ethresh = 1e-4
        lmax = 2
        rc = 4
        kappa, K1, K2, K3 = setup_ewald_parameters(rc, ethresh, box)

        displacement_fn, shift_fn = space.periodic_general(
            box, fractional_coordinates=False
        )
        neighbor_list_fn = partition.neighbor_list(
            displacement_fn, box, rc, 0, format=partition.OrderedSparse
        )
        nbr = neighbor_list_fn.allocate(positions)
        pairs = nbr.idx.T

        pme_force = ADMPPmeForce(
            box, axis_types, axis_indices, covalent_map, rc, ethresh, lmax
        )
        
        pme_force.update_env("kappa", 0.6570)
        # pme_force.update_env("kappa", kappa)
        
        E, F = pme_force.get_forces(positions, box, pairs, Q_local, mScales)

        construct_local_frame_fn = generate_construct_local_frames(
            axis_types, axis_indices
        )
        pme_order = 6
        pme_recip_fn = generate_pme_recip(
            Ck_1, kappa, False, pme_order, K1, K2, K3, lmax
        )
        
        H = Hamiltonian('examples/openmm_api/forcefield.xml')

        generators = H.getGenerators()
        app.Topology.loadBondDefinitions("examples/openmm_api/residues.xml")
        pdb = app.PDBFile("/share/home/jichen/work/ADMP/water1024.pdb")
        rc = 4.0
        potentials = H.createPotential(pdb.topology, nonbondedCutoff=rc*unit.angstrom)
        pot_disp = potentials[0]
        pot_pme = potentials[1]

        positions_ = jnp.array(pdb.positions._value) * 10
        a, b, c = pdb.topology.getPeriodicBoxVectors()
        box_ = jnp.array([a._value, b._value, c._value]) * 10        
        mScales = generators[0].params['mScales']
        # positions, box, pairs, Q_local, mScales
        E_, F_ = pot_pme(positions_, box_, pairs, generators[1].params)

        E, F = pme_force.get_forces(positions_, box_, pairs, generators[1].params['Q_local'], generators[1].params['mScales'])
        
        # assert E_ == E
        # npt.assert_allclose(F_, F)
