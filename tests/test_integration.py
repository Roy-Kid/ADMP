from time import time

import jax.numpy as jnp
import numpy as np
import pytest
from admp.multipole import rot_local2global
from admp.pme import ADMPPmeForce, pme_real, pme_self, setup_ewald_parameters
from admp.recip import Ck_1, generate_pme_recip
from admp.spatial import generate_construct_local_frames
from jax._src.api import value_and_grad
from jax_md import partition, space


class TestResultWithMPID:
    
    def test_PmeForce(self, water):
        
        expected_energy = {
            'water1': 1234,
            'water2': 2345
        }
        
        mScales = jnp.array([0.0, 0.0, 0.0, 1.0, 1.0])
        pScales = jnp.array([0.0, 0.0, 0.0, 1.0, 1.0])
        dScales = jnp.array([0.0, 0.0, 0.0, 1.0, 1.0])

        rc = 4  # in Angstrom
        ethresh = 1e-4
        lmax = 2
        
        nwater, (serials, names, resNames, resSeqs, positions, box, charges, Q_local, axis_types, axis_indices, covalent_map) = water
        
        displacement_fn, shift_fn = space.periodic_general(box, fractional_coordinates=False)
        neighbor_list_fn = partition.neighbor_list(displacement_fn, box, rc, 0, format=partition.OrderedSparse)
        nbr = neighbor_list_fn.allocate(positions)
        pairs = nbr.idx.T        
        
        pme_force = ADMPPmeForce(box, axis_types, axis_indices, covalent_map, rc, ethresh, lmax)
        pme_force.update_env('kappa', 0.657065221219616)
        
        ck0 = time()
        E, F = pme_force.get_forces(positions, box, pairs, Q_local, mScales, pScales, dScales)
        ck1 = time()
        E, F = pme_force.get_forces(positions, box, pairs, Q_local, mScales, pScales, dScales)
        ck2 = time()
        assert ck2 - ck1 < ck1 - ck0, RuntimeError(f'After JIT, the running time should be much less than the first time. First time with JIT is {ck1 - ck0} and second after JIT is {ck1 - ck0}')
        
    def test_pme_real(self, water):
        
        mScales = jnp.array([0.0, 0.0, 0.0, 1.0, 1.0])
        pScales = jnp.array([0.0, 0.0, 0.0, 1.0, 1.0])
        dScales = jnp.array([0.0, 0.0, 0.0, 1.0, 1.0])

        rc = 4  # in Angstrom
        ethresh = 1e-4
        lmax = 2
        kappa = 0.657065221219616
        
        nwater, (serials, names, resNames, resSeqs, positions, box, charges, Q_local, axis_types, axis_indices, covalent_map) = water
        
        displacement_fn, shift_fn = space.periodic_general(box, fractional_coordinates=False)
        neighbor_list_fn = partition.neighbor_list(displacement_fn, box, rc, 0, format=partition.OrderedSparse)
        nbr = neighbor_list_fn.allocate(positions)
        pairs = nbr.idx.T        

        construct_local_frame_fn = generate_construct_local_frames(axis_types, axis_indices)

        def mock_energy_pme(positions, box, pairs, Q_local, mScales, pScales, dScales):
            local_frames = construct_local_frame_fn(positions, box)
            Q_global = rot_local2global(Q_local, local_frames, lmax)
            ene_real = pme_real(positions, box, pairs, Q_global, mScales, covalent_map, kappa, lmax)
            return ene_real
        
        e, f = value_and_grad(mock_energy_pme)(positions, box, pairs, Q_local, mScales, pScales, dScales)
            
    def test_pme_self(self, water):
        
        mScales = jnp.array([0.0, 0.0, 0.0, 1.0, 1.0])
        pScales = jnp.array([0.0, 0.0, 0.0, 1.0, 1.0])
        dScales = jnp.array([0.0, 0.0, 0.0, 1.0, 1.0])

        rc = 4  # in Angstrom
        ethresh = 1e-4
        lmax = 2
        kappa = 0.657065221219616
        
        nwater, (serials, names, resNames, resSeqs, positions, box, charges, Q_local, axis_types, axis_indices, covalent_map) = water
        
        displacement_fn, shift_fn = space.periodic_general(box, fractional_coordinates=False)
        neighbor_list_fn = partition.neighbor_list(displacement_fn, box, rc, 0, format=partition.OrderedSparse)
        nbr = neighbor_list_fn.allocate(positions)
        pairs = nbr.idx.T        
        
        construct_local_frame_fn = generate_construct_local_frames(axis_types, axis_indices)

        def mock_energy_pme(positions, box, pairs, Q_local, mScales, pScales, dScales):
            local_frames = construct_local_frame_fn(positions, box)
            Q_global = rot_local2global(Q_local, local_frames, lmax)
            ene_real = pme_self(Q_local, kappa, lmax)
            return ene_real
        
        e, f = value_and_grad(mock_energy_pme)(positions, box, pairs, Q_local, mScales, pScales, dScales)
        
    def test_pme_reci(self, water):
        
        mScales = jnp.array([0.0, 0.0, 0.0, 1.0, 1.0])
        pScales = jnp.array([0.0, 0.0, 0.0, 1.0, 1.0])
        dScales = jnp.array([0.0, 0.0, 0.0, 1.0, 1.0])

        rc = 4  # in Angstrom
        ethresh = 1e-4
        lmax = 2
        kappa = 0.657065221219616
        pme_order = 6
        
        nwater, (serials, names, resNames, resSeqs, positions, box, charges, Q_local, axis_types, axis_indices, covalent_map) = water
        
        kappa, K1, K2, K3 = setup_ewald_parameters(rc, ethresh, box)
        
        displacement_fn, shift_fn = space.periodic_general(box, fractional_coordinates=False)
        neighbor_list_fn = partition.neighbor_list(displacement_fn, box, rc, 0, format=partition.OrderedSparse)
        nbr = neighbor_list_fn.allocate(positions)
        pairs = nbr.idx.T        
        
        construct_local_frame_fn = generate_construct_local_frames(axis_types, axis_indices)
        
        pme_recip_fn = generate_pme_recip(Ck_1, kappa, False, pme_order, K1, K2, K3, lmax)

        def mock_energy_pme(positions, box, pairs,
        Q_local, mScales, pScales, dScales, covalent_map, 
        construct_local_frame_fn, pme_recip_fn, kappa, K1, K2, K3, lmax):
            local_frames = construct_local_frame_fn(positions, box)
            Q_global = rot_local2global(Q_local, local_frames, lmax)
            ene_recip = pme_recip_fn(positions, box, Q_global)
            return ene_recip
        
        e, f = value_and_grad(mock_energy_pme)(positions, box, pairs,
        Q_local, mScales, pScales, dScales, covalent_map, 
        construct_local_frame_fn, pme_recip_fn, kappa, K1, K2, K3, lmax)        
