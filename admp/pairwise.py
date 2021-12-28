#!/usr/bin/env python
import sys
import jax
from jax import vmap
from admp.settings import *
from admp.spatial import *

# for debug
from jax_md import partition, space
from admp.parser import *
from admp.multipole import *
from jax import grad, value_and_grad
from admp.pme import *


def generate_pairwise_interaction(pair_int_kernel, covalent_map, static_args):

    def pair_int(positions, box, pairs, mScales, *atomic_params):
        pairs =  pairs[pairs[:, 0] < pairs[:, 1]]
        ri = positions[pairs[:, 0]]
        rj = positions[pairs[:, 1]]
        nbonds = covalent_map[pairs[:, 0], pairs[:, 1]]
        mscales = mScales[nbonds-1]
        box_inv = jnp.linalg.inv(box)
        dr = ri - rj
        dr = pbc_shift(dr, box, box_inv)
        dr = jnp.linalg.norm(dr, axis=1)

        pair_params = []
        for i, param in enumerate(atomic_params):
            pair_params.append(param[pairs[:, 0]])
            pair_params.append(param[pairs[:, 1]])

        energy = jnp.sum(pair_int_kernel(dr, mscales, *pair_params))
        return energy

    return pair_int



# below is the validation code
# if __name__ == '__main__':
#     pdb = str(sys.argv[1])
#     xml = 'mpidwater.xml'
#     pdbinfo = read_pdb(pdb)
#     serials = pdbinfo['serials']
#     names = pdbinfo['names']
#     resNames = pdbinfo['resNames']
#     resSeqs = pdbinfo['resSeqs']
#     positions = pdbinfo['positions']
#     box = pdbinfo['box'] # a, b, c, α, β, γ
#     charges = pdbinfo['charges']
#     positions = jnp.asarray(positions)
#     lx, ly, lz, _, _, _ = box
#     box = jnp.eye(3)*jnp.array([lx, ly, lz])

#     mScales = jnp.array([0.0, 0.0, 0.0, 1.0, 1.0])
#     pScales = jnp.array([0.0, 0.0, 0.0, 1.0, 1.0])
#     dScales = jnp.array([0.0, 0.0, 0.0, 1.0, 1.0])

#     rc = 4  # in Angstrom
#     ethresh = 1e-4

#     n_atoms = len(serials)

#     atomTemplate, residueTemplate = read_xml(xml)
#     atomDicts, residueDicts = init_residues(serials, names, resNames, resSeqs, positions, charges, atomTemplate, residueTemplate)

#     covalent_map = assemble_covalent(residueDicts, n_atoms)
#     displacement_fn, shift_fn = space.periodic_general(box, fractional_coordinates=False)
#     neighbor_list_fn = partition.neighbor_list(displacement_fn, box, rc, 0, format=partition.OrderedSparse)
#     nbr = neighbor_list_fn.allocate(positions)
#     pairs = nbr.idx.T

#     pmax = 10
#     kappa, K1, K2, K3 = setup_ewald_parameters(rc, ethresh, box)
#     kappa = 0.657065221219616

#     # construct the C list
#     c_list = np.zeros((3,n_atoms))
#     nmol=int(n_atoms/3)
#     for i in range(nmol):
#         a = i*3
#         b = i*3+1
#         c = i*3+2
#         c_list[0][a]=37.19677405
#         c_list[0][b]=7.6111103
#         c_list[0][c]=7.6111103
#         c_list[1][a]=85.26810658
#         c_list[1][b]=11.90220148
#         c_list[1][c]=11.90220148
#         c_list[2][a]=134.44874488
#         c_list[2][b]=15.05074749
#         c_list[2][c]=15.05074749
#     c_list = jnp.array(c_list)

#     @partial(vmap, in_axes=(0, 0, 0, 0), out_axes=(0))
#     @jit_condition(static_argnums=())
#     def disp6_pme_real_kernel(dr, m, ci, cj):
#         # unpack static arguments
#         kappa = static_args['kappa']
#         # calculate distance
#         dr2 = dr ** 2
#         dr6 = dr2 ** 3
#         # do calculation
#         x2 = kappa**2 * dr2
#         exp_x2 = jnp.exp(-x2)
#         x4 = x2 * x2
#         g = (1 + x2 + 0.5*x4) * exp_x2
#         return (m + g - 1) * ci * cj / dr6
   
#     static_args = {'kappa': kappa}
#     disp6_pme_real = generate_pairwise_interaction(disp6_pme_real_kernel, covalent_map, static_args)
#     print(disp6_pme_real(positions, box, pairs, mScales, c_list[0, :]))
    
