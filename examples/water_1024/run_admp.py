#!/usr/bin/env python
import sys
import numpy as np
import jax.numpy as jnp
from jax import grad, value_and_grad
from admp.multipole import *
from admp.parser import *
from admp.pme import *
from admp.disp_pme import *


# below is the validation code
if __name__ == '__main__':
    pdb = str(sys.argv[1])
    xml = 'mpidwater.xml'
    pdbinfo = read_pdb(pdb)
    serials = pdbinfo['serials']
    names = pdbinfo['names']
    resNames = pdbinfo['resNames']
    resSeqs = pdbinfo['resSeqs']
    positions = pdbinfo['positions']
    box = pdbinfo['box'] # a, b, c, α, β, γ
    charges = pdbinfo['charges']
    positions = jnp.asarray(positions)
    lx, ly, lz, _, _, _ = box
    box = jnp.eye(3)*jnp.array([lx, ly, lz])

    mScales = jnp.array([0.0, 0.0, 0.0, 1.0, 1.0])
    pScales = jnp.array([0.0, 0.0, 0.0, 1.0, 1.0])
    dScales = jnp.array([0.0, 0.0, 0.0, 1.0, 1.0])

    rc = 4  # in Angstrom
    ethresh = 1e-4

    n_atoms = len(serials)

    atomTemplate, residueTemplate = read_xml(xml)
    atomDicts, residueDicts = init_residues(serials, names, resNames, resSeqs, positions, charges, atomTemplate, residueTemplate)

    Q = np.vstack(
        [(atom.c0, atom.dX*10, atom.dY*10, atom.dZ*10, atom.qXX*300, atom.qYY*300, atom.qZZ*300, atom.qXY*300, atom.qXZ*300, atom.qYZ*300) for atom in atomDicts.values()]
    )
    Q = jnp.array(Q)
    Q_local = convert_cart2harm(Q, 2)
    axis_type = np.array(
        [atom.axisType for atom in atomDicts.values()]
    )
    axis_indices = np.vstack(
        [atom.axis_indices for atom in atomDicts.values()]
    )
    covalent_map = assemble_covalent(residueDicts, n_atoms)

    
    lmax = 2
    pmax = 10

    # construct the C list
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
    c_list = jnp.array(c_list.T)


    # Finish data preparation
    # -------------------------------------------------------------------------------------
    # parameters should be ready: 
    # geometric variables: positions, box
    # atomic parameters: Q_local, c_list
    # topological parameters: covalent_map, mScales, pScales, dScales
    # general force field setting parameters: rc, ethresh, lmax, pmax


    # get neighbor list using jax_md
    displacement_fn, shift_fn = space.periodic_general(box, fractional_coordinates=False)
    neighbor_list_fn = partition.neighbor_list(displacement_fn, box, rc, 0, format=partition.OrderedSparse)
    nbr = neighbor_list_fn.allocate(positions)
    pairs = nbr.idx.T


    # electrostatic
    pme_force = ADMPPmeForce(box, axis_type, axis_indices, covalent_map, rc, ethresh, lmax)
    pme_force.update_env('kappa', 0.657065221219616)
    E, F = pme_force.get_forces(positions, box, pairs, Q_local, mScales, pScales, dScales)
    print('Electrostatic Energy (kJ/mol)')
    E, F = pme_force.get_forces(positions, box, pairs, Q_local, mScales, pScales, dScales)
    print(E)


    # dispersion
    disp_pme_force = ADMPDispPmeForce(box, covalent_map, rc, ethresh, pmax)
    disp_pme_force.update_env('kappa', 0.657065221219616)
    E, F = disp_pme_force.get_forces(positions, box, pairs, c_list, mScales)
    print('Dispersion Energy (kJ/mol)')
    E, F = disp_pme_force.get_forces(positions, box, pairs, c_list, mScales)
    print(E)
   

