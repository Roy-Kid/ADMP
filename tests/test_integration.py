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


@contextmanager
def timer(type, job):
    try:
        start = time()
        yield
        end = time()
    finally:
        print(f"{end-start:.4f}s {type}\tpme_force\t[{job}]")


class TestResultWithMPID:
    @pytest.fixture(scope="class", name="pme_force_fixture")
    def test_jit_pme_force(self, water):

        expected_energy = {"water1": 1234, "water2": 2345}
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
                Q_local,
                axis_types,
                axis_indices,
                covalent_map,
            ),
        ) = water

        mScales = jnp.array([0.0, 0.0, 0.0, 1.0, 1.0])
        pScales = jnp.array([0.0, 0.0, 0.0, 1.0, 1.0])
        dScales = jnp.array([0.0, 0.0, 0.0, 1.0, 1.0])

        rc = 4  # in Angstrom
        ethresh = 1e-4
        lmax = 2
        kappa, K1, K2, K3 = setup_ewald_parameters(rc, ethresh, box)
        kappa = 0.657065221219616

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
        pme_force.update_env("kappa", 0.657065221219616)

        construct_local_frame_fn = generate_construct_local_frames(
            axis_types, axis_indices
        )
        pme_order = 6
        pme_recip_fn = generate_pme_recip(
            Ck_1, kappa, False, pme_order, K1, K2, K3, lmax
        )

        def mock_pme_real(positions, box, pairs, Q_local, mScales, pScales, dScales):
            local_frames = construct_local_frame_fn(positions, box)
            Q_global = rot_local2global(Q_local, local_frames, lmax)
            ene_real = pme_real(
                positions, box, pairs, Q_global, mScales, covalent_map, kappa, lmax
            )
            return ene_real

        def mock_pme_self(positions, box, pairs, Q_local, mScales, pScales, dScales):
            local_frames = construct_local_frame_fn(positions, box)
            Q_global = rot_local2global(Q_local, local_frames, lmax)
            ene_self = pme_self(Q_local, kappa, lmax)
            return ene_self

        def mock_pme_reci(
            positions,
            box,
            pairs,
            Q_local,
            mScales,
            pScales,
            dScales,
            covalent_map,
            construct_local_frame_fn,
            pme_recip_fn,
            kappa,
            K1,
            K2,
            K3,
            lmax,
        ):
            local_frames = construct_local_frame_fn(positions, box)
            Q_global = rot_local2global(Q_local, local_frames, lmax)
            ene_recip = pme_recip_fn(positions, box, Q_global)
            return ene_recip

        print("\n=== start jitting ===\n")
        # natrue jit
        with timer('jitting', f"water{nwater}"):
            E, F = pme_force.get_forces(
                positions, box, pairs, Q_local, mScales, pScales, dScales
            )
            E.block_until_ready()

        with timer('jitting', f"water{nwater}"):
            E, F = value_and_grad(mock_pme_real)(
                positions, box, pairs, Q_local, mScales, pScales, dScales
            )
            E.block_until_ready()

        with timer('jitting', f"water{nwater}"):
            E, F = value_and_grad(mock_pme_self)(
                positions, box, pairs, Q_local, mScales, pScales, dScales
            )
            E.block_until_ready()

        with timer('jitting', f"water{nwater}"):
            E, F = value_and_grad(mock_pme_reci)(
                positions,
                box,
                pairs,
                Q_local,
                mScales,
                pScales,
                dScales,
                covalent_map,
                construct_local_frame_fn,
                pme_recip_fn,
                kappa,
                K1,
                K2,
                K3,
                lmax,
            )
            E.block_until_ready()

        jitted_forces = pme_force.get_forces
        jitted_real = value_and_grad(mock_pme_real)
        jitted_self = value_and_grad(mock_pme_self)
        jitted_reci = value_and_grad(mock_pme_reci)

        yield nwater, jitted_forces, jitted_real, jitted_self, jitted_reci, positions, box, pairs, Q_local, mScales, pScales, dScales, kappa, K1, K2, K3, axis_types, axis_indices, covalent_map, construct_local_frame_fn, pme_order, pme_recip_fn

    def test_jitted_pme_force(self, pme_force_fixture):
        (
            nwater,
            jitted_forces,
            jitted_real,
            jitted_self,
            jitted_reci,
            positions,
            box,
            pairs,
            Q_local,
            mScales,
            pScales,
            dScales,
            *_
        ) = pme_force_fixture
        with timer('compute', f"water{nwater}"):
            E, F = jitted_forces(
                positions, box, pairs, Q_local, mScales, pScales, dScales
            )
            E.block_until_ready()

    def test_pme_real(self, pme_force_fixture):

        (
            nwater,
            jitted_forces,
            jitted_real,
            jitted_self,
            jitted_reci,
            positions,
            box,
            pairs,
            Q_local,
            mScales,
            pScales,
            dScales,
            *_
        ) = pme_force_fixture
        with timer('compute', f"water{nwater}"):
            E, F = jitted_real(
                positions, box, pairs, Q_local, mScales, pScales, dScales
            )
            E.block_until_ready()

    def test_pme_self(self, pme_force_fixture):

        (
            nwater,
            jitted_forces,
            jitted_real,
            jitted_self,
            jitted_reci,
            positions,
            box,
            pairs,
            Q_local,
            mScales,
            pScales,
            dScales,
            *_
        ) = pme_force_fixture

        with timer('compute', f"water{nwater}"):
            E, F = jitted_self(
                positions, box, pairs, Q_local, mScales, pScales, dScales
            )
            E.block_until_ready()

    def test_pme_reci(self, water, pme_force_fixture):

        (
            nwater,
            jitted_forces,
            jitted_real,
            jitted_self,
            jitted_reci,
            positions,
            box,
            pairs,
            Q_local,
            mScales,
            pScales,
            dScales,
            kappa,
            K1,
            K2,
            K3,
            axis_types,
            axis_indices,
            covalent_map,
            construct_local_frame_fn,
            pme_order,
            pme_recip_fn,
        ) = pme_force_fixture

        lmax = 2
        with timer('compute', f"water{nwater}"):
            E, F = jitted_reci(
                positions,
                box,
                pairs,
                Q_local,
                mScales,
                pScales,
                dScales,
                covalent_map,
                construct_local_frame_fn,
                pme_recip_fn,
                kappa,
                K1,
                K2,
                K3,
                lmax,
            )
            E.block_until_ready()

