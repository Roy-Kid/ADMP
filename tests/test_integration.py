import pytest
import jax.numpy as jnp
import numpy as np
from jax_md import space, partition
from jax import value_and_grad, jit
from admp.multipole import convert_cart2harm, rot_local2global
import numpy.testing as npt
from admp.spatial import generate_construct_local_frames
from admp.pme import pme_real, pme_self, setup_ewald_parameters
from admp.recip import generate_pme_recip, Ck_1, Ck_6, Ck_8, Ck_10


class TestResultsWithMPID:
    # jichen:
    # test every part of pme equals to MPID
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
        kappa, K1, K2, K3 = setup_ewald_parameters(rc, ethresh, box)
        print(f'{kappa=} {K1=} {K2=} {K3=}')
        K1 = 157
        K2 = 157
        K3 = 157
        kappa = 0.657065221219616
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

    @pytest.mark.parametrize(
        "water, expected_e",
        [
            # ("water1", 841.4397355),
            # ("water2", 1682.954775),
            # ("water4", 3365.931692),
            # ("water8", 6731.664416),
            # ("water256", 215423.9477),
            # ("water512", 430905.7252),
            ("water1024", 861884.4751),
        ],
    )
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

        energy_force_pme = value_and_grad(mock_energy_pme)
        e, f = energy_force_pme(
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

    @pytest.mark.parametrize(
        "water, expected_e",
        [
            # ("water1", -878.794685),
            # ("water2", -1757.58937),
            # ("water4", -3515.17874),
            # ("water8", -7030.35748),
            # ("water256", -224971.4394),
            # ("water512", -449942.8787),
            ("water1024", -899885.7575),
        ],
    )
    def test_self_energy_force(self, water, expected_e, request):
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

        def mock_energy_pme(
            positions,
            box,
            Q_local,
            mScales,
            pScales,
            dScales,
            covalent_map,
            kappa,
            K1,
            K2,
            K3,
            lmax,
        ):

            e = pme_self(Q_local, kappa, lmax)
            return e

        energy_force_pme = value_and_grad(mock_energy_pme)
        e, f = energy_force_pme(
            positions,
            box,
            Q_local,
            mScales,
            pScales,
            dScales,
            covalent_map,
            kappa,
            K1,
            K2,
            K3,
            lmax,
        )
        npt.assert_allclose(e, expected_e)

    @pytest.mark.parametrize(
        "water, expected_e",
        [
            ("water1", 37.35318637),
            ("water2", 74.64692248),
            ("water4", 149.1969829),
            ("water8", 298.9894403),
            ("water256", 9561.106475),
            ("water512", 19110.99589),
            ("water1024", 38149.4628),
        ],
    )
    def test_reci_energy_force(self, water, expected_e, request):
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
        construct_local_frames_fn = generate_construct_local_frames(
            axis_types, axis_indices
        )
        local_frames = construct_local_frames_fn(positions, box)
        Q_global = rot_local2global(Q_local, local_frames, lmax)

        pme_order = 6
        energy_force_pme_recip = value_and_grad(
            generate_pme_recip(Ck_1, kappa, False, pme_order, K1, K2, K3, lmax)
        )
        e, f = energy_force_pme_recip(positions, box, Q_global)

        # construct the C list
        n_atoms = len(positions)
        c_list = np.zeros((3,n_atoms))
        nmol=int(n_atoms/3)
        for i in range(nmol):
            a = i*3
            b = i*3+1
            c = i*3+2
            c_list[0][a]=37.19677405
            c_list[0][b]=7.6111103
            c_list[0][c]=7.6111103
            c_list[1][a]=85.26810658
            c_list[1][b]=11.90220148
            c_list[1][c]=11.90220148
            c_list[2][a]=134.44874488
            c_list[2][b]=15.05074749
            c_list[2][c]=15.05074749
        energy_force_d6_recip = value_and_grad(generate_pme_recip(Ck_6, kappa, True, pme_order, K1, K2, K3, 0))
        energy_force_d8_recip = value_and_grad(generate_pme_recip(Ck_8, kappa, True, pme_order, K1, K2, K3, 0))
        energy_force_d10_recip = value_and_grad(generate_pme_recip(Ck_10, kappa, True, pme_order, K1, K2, K3, 0))
        E6, F6 = energy_force_d6_recip(positions, box, c_list[0, :, jnp.newaxis])
        E8, F6 = energy_force_d8_recip(positions, box, c_list[1, :, jnp.newaxis])
        E10, F10 = energy_force_d10_recip(positions, box, c_list[2, :, jnp.newaxis])
        
        npt.assert_allclose(e, expected_e)

    @pytest.mark.parametrize("water, expected_e", [            ("water1", 37.35318637),
            ("water2", 74.64692248),
            ("water4", 149.1969829),
            ("water8", 298.9894403),
            ("water256", 9561.106475),
            ("water512", 19110.99589),
            ("water1024", 38149.4628),])
    def test_disp_real_energy(self, water, expected_e, request):
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
        pass

    @pytest.mark.parametrize("water, expected_e", [            ("water1", 37.35318637),
            ("water2", 74.64692248),
            ("water4", 149.1969829),
            ("water8", 298.9894403),
            ("water256", 9561.106475),
            ("water512", 19110.99589),
            ("water1024", 38149.4628),])
    def test_disp_self_energy(self, water, expected_e, request):
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
        pass