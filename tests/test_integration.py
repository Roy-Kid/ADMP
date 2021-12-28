import pytest
import jax.numpy as jnp
from jax_md import space, partition
from jax import value_and_grad, jit
from admp.multipole import convert_cart2harm, rot_local2global
import numpy.testing as npt
from admp.spatial import generate_construct_local_frames
from admp.pme import energy_pme, pme_real


class TestResultsWithMPID:
    
    def convertedWater(self, water):
        positions, box, atomDicts, axis_types, axis_indices, covalent_map = water
        Q = jnp.vstack(
            [
                (
                    atom.c0,
                    atom.dX * 10,
                    atom.dY * 10,
                    atom.dZ * 10,
                    atom.qXX * 300,
                    atom.qYY * 300,
                    atom.qZZ * 300,
                    atom.qXY * 300,
                    atom.qXZ * 300,
                    atom.qYZ * 300,
                )
                for atom in atomDicts.values()
            ]
        )
        Q_local = convert_cart2harm(Q, 2)
        rc = 4
        ethresh = 1e-4
        lmax = 2
        kappa = 0.657065221219616
        K1 = 97
        K2 = 97
        K3 = 97
        mScales = jnp.array([0.0, 0.0, 0.0, 1.0, 1.0])
        pScales = jnp.array([0.0, 0.0, 0.0, 1.0, 1.0])
        dScales = jnp.array([0.0, 0.0, 0.0, 1.0, 1.0])
        return (
            positions,
            Q_local,
            box,
            atomDicts,
            axis_types,
            axis_indices,
            covalent_map,
            rc,
            lmax,
            kappa,
            K1,
            K2,
            K3,
            mScales,
            pScales,
            dScales,
        )

    @pytest.mark.parametrize('water, expected_e', [('water1', 841.4659601
), ('water2', 1683.038432
)])
    def test_real_energy_force(self, water, expected_e, request):
        # for debugging
        # Consistent with MPID
        (
            positions,
            Q_local,
            box,
            atomDicts,
            axis_types,
            axis_indices,
            covalent_map,
            rc,
            lmax,
            kappa,
            K1,
            K2,
            K3,
            mScales,
            pScales,
            dScales,
        ) = self.convertedWater(request.getfixturevalue(water))
        displacement_fn, shift_fn = space.periodic_general(
            box, fractional_coordinates=False
        )
        neighbor_list_fn = partition.neighbor_list(
            displacement_fn, box, rc, 0, format=partition.OrderedSparse
        )
        nbr = neighbor_list_fn.allocate(positions)
        pairs = nbr.idx.T
        construct_local_frame_fn = generate_construct_local_frames(
            axis_types, axis_indices
        )

        def mock_energy_pme(
            positions,
            box,
            pairs,
            Q_local,
            mScales,
            pScales,
            dScales,
            covalent_map,
            construct_local_frame_fn,
            kappa,
            K1,
            K2,
            K3,
            lmax,
        ):

            local_frames = construct_local_frame_fn(positions, box)
            Q_global = rot_local2global(Q_local, local_frames, lmax)
            e = pme_real(
                positions, box, pairs, Q_global, mScales, covalent_map, kappa, lmax
            )
            return e

        e, f = value_and_grad(mock_energy_pme)(
            positions,
            box,
            pairs,
            Q_local,
            mScales,
            pScales,
            dScales,
            covalent_map,
            construct_local_frame_fn,
            kappa,
            K1,
            K2,
            K3,
            lmax,
        )

        npt.assert_allclose(e, expected_e)
        # npt.assert_allclose(f, expected_f)

