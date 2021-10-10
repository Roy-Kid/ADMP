#!/usr/bin/env python
import numpy as np

np.random.seed(100)

r1 = np.zeros(3)
r2 = np.ones(3)
# box = np.array([
#      [19.0, 0.00, 0.00],
#      [0.0, 19.0, 0.0],
#      [0.0, 0.0, 19.0]
#      ])
box = np.array([
     [30.0, 0.00, 0.00],
     [0.0, 40.0, 0.0],
     [0.0, 0.0, 35.0]
     ])
# box = np.array([
#      [30.0, 0.00, 0.00],
#      [-17.320508075688775, 30.0, 0.0],
#      [0.0, 0.0, 35.0]
#      ])
box_inv = np.linalg.inv(box)

# construct inputs
n_atoms = 4000
r_cut = 4.0
spositions = np.random.random((n_atoms, 3))
positions = spositions.dot(box)

################################
# check data in the cell list
################################
# print('*****, printing some randomly selected elements in neighbour list')
# print('---cell_list components---')
# print(celllist.shape)
# i = 5
# j = 6
# k = 6
# na, nb, nc, nmax = dimensions
# print(celllist[i, j, k, :])
# ib = nmax * (k + j*nc + i*nc*nb)
# ie = nmax * (k + j*nc + i*nc*nb + 1)
# print(celllist_1d[4+ib:4+ie])
# # print('---cell_index_map_test---')
# # print(positions[392])
# # print(box[0]/dimensions[0], box[1]/dimensions[1], box[2]/dimensions[2])
# # print(cell_index_map[392])
# print('---neighbours---')
# i0 = 1
# i1 = 1809
# print('indices')
# print(i0, i1)
# print(nbs[i0])
# print(np.sqrt(distances2)[i0,:])
# print(nbs[i1])
# # print(neighbour_list.find_neighbours_for_atom(positions.T, celllist_1d, i0, cell_index_map.T, r_cut, box.T, box_inv.T))
# print('positions')
# r0 = positions[i0]
# r1 = positions[i1]
# print(r0);
# print(r1);
# print('distances')
# print(spatial.distance_pbc(r0, r1, box.T, box_inv.T, 0))
# print('cell_index_map')
# print(cell_index_map[i0])
# print(cell_index_map[i1])

def construct_nblist(positions, box, rc):
    # Some constants
    CELL_LIST_BUFFER = 80
    MAX_N_NEIGHBOURS = 600

    # construct cells
    box_inv = np.linalg.inv(box)
    # len_abc_star = np.linalg.norm(box_inv, axis=0)
    len_abc_star = np.sqrt(np.sum(box_inv*box_inv, axis=0))
    h_box = 1 / len_abc_star # the "heights" of the box
    n_cells = np.floor(h_box/rc).astype(np.int32)
    n_totc = np.prod(n_cells)
    n_atoms = len(positions)
    density = n_atoms / np.prod(n_cells)
    n_max = int(np.floor(density)) + CELL_LIST_BUFFER
    cell = np.diag(1./n_cells).dot(box)
    cell_inv = np.linalg.inv(cell)

    upositions = positions.dot(cell_inv)
    indices = np.floor(upositions).astype(np.int32)
    # for safty, do pbc shift
    indices[:, 0] = indices[:, 0] % n_cells[0]
    indices[:, 1] = indices[:, 1] % n_cells[1]
    indices[:, 2] = indices[:, 2] % n_cells[2]
    cell_list = np.full((n_cells[0], n_cells[1], n_cells[2], n_max), -1)
    counts = np.full((n_cells[0], n_cells[1], n_cells[2]), 0)
    for i_atom in range(n_atoms):
        ia, ib, ic = indices[i_atom]
        cell_list[ia, ib, ic, counts[ia, ib, ic]] = i_atom
        counts[ia, ib, ic] += 1

    # cell adjacency
    na, nb, nc = n_cells
    cell_list = cell_list.reshape((n_totc, n_max))
    counts = counts.reshape(n_totc)
   
    rc2 = rc * rc
    nbs = np.empty((n_atoms, MAX_N_NEIGHBOURS), dtype=np.int32)
    n_nbs = np.zeros(n_atoms, dtype=np.int32)
    distances2 = np.empty((n_atoms, MAX_N_NEIGHBOURS))
    dr_vecs = np.empty((n_atoms, MAX_N_NEIGHBOURS, 3))
    icells_3d = np.empty((n_totc, 3))
    for ic in range(n_totc):
        icells_3d[ic] = np.array([ic // (nb*nc), (ic // nc) % nb, ic % nc], dtype=np.int32)
    for ic in range(n_totc):
        ic_3d = icells_3d[ic]
        if counts[ic] == 0:
            continue
        for jc in range(ic, n_totc):
            if counts[jc] == 0:
                continue
            jc_3d = icells_3d[jc]
            di_cell = jc_3d - ic_3d
            di_cell -= (di_cell + n_cells//2) // n_cells * n_cells
            max_di = np.max(np.abs(di_cell))
            # two cells are not neighboring
            if max_di > 1:
                continue
            indices_i = cell_list[ic, 0:counts[ic]]
            indices_j = cell_list[jc, 0:counts[jc]]
            indices_i = np.repeat(indices_i, counts[jc])
            indices_j = np.repeat(indices_j, counts[ic]).reshape(-1, counts[ic]).T.flatten()
            if ic == jc:
                ifilter = (indices_j > indices_i)
                indices_i = indices_i[ifilter]
                indices_j = indices_j[ifilter]
            ri = positions[indices_i]
            rj = positions[indices_j]
            dr = rj - ri
            ds = np.dot(dr, box_inv)
            ds -= np.floor(ds+0.5)
            dr = np.dot(ds, box)
            drdr = np.sum(dr*dr, axis=1)
            for ii in np.where(drdr < rc2)[0]:
                i = indices_i[ii]
                j = indices_j[ii]
                nbs[i, n_nbs[i]] = j
                nbs[j, n_nbs[j]] = i
                distances2[i, n_nbs[i]] = drdr[ii]
                distances2[j, n_nbs[j]] = drdr[ii]
                dr_vecs[i, n_nbs[i], :] = dr[ii]
                dr_vecs[j, n_nbs[j], :] = -dr[ii]
                n_nbs[i] += 1
                n_nbs[j] += 1
       
    return nbs, n_nbs, distances2, dr_vecs

nbs, n_nbs, distances2, dr_vecs = construct_nblist(positions, box, r_cut)
# construct_nblist(positions, box, r_cut)


################################
# systematic test neighbour list
################################
# flag = 0
# print('*****, Start testing nbs and distances2...')
# for i_atom0, neighbours in enumerate(nbs):
#     neighbours = neighbours[0:n_nbs[i_atom0]]
#     r0 = positions[i_atom0]
#     print('Testing particle', i_atom0)
#     for i_atom1 in range(n_atoms):
#         if i_atom1 == i_atom0:
#             continue 
#         r1 = positions[i_atom1]
#         dr = spatial.distance_pbc(r0, r1, box.T, box_inv.T, 0)
#         if dr >= r_cut and i_atom1 in neighbours:
#             print('ERROR: Wrong interacting pair:', i_atom0, i_atom1, dr)
#             flag = 1
#         elif dr < r_cut:
#             index = np.where(neighbours==i_atom1)[0]
#             if len(index) == 0:
#                 print('ERROR: Missing interacting pair:', i_atom0, i_atom1, dr)
#                 flag = 1
#             elif len(index) > 1:
#                 print('ERROR: Duplication in neighbour list:', i_atom0, i_atom1, dr)
#                 flag = 1
#             else:
#                 if abs(np.sqrt(distances2[i_atom0, index[0]]) - dr) > 1e-6:
#                     print('ERROR: Distances mismatch:', i_atom0, i_atom1, dr, np.sqrt(distances2[i_atom0, index[0]]))
#                     flag = 1
# if flag == 0:
#    print('Neighbour list checks out.')
        

# print('---number of neighbours---')
# print('max', 'min', 'aver')
# n_nbs = ((nbs>0)>0).sum(axis=1)
# print(n_nbs.max(), n_nbs.min(), np.average(n_nbs))
