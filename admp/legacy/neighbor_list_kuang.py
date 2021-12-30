#!/usr/bin/env python

# This is one of the many versions of neighbor list we tried

def construct_nblist(positions, box, rc):
    jax.lax.stop_gradient(box)
    jax.lax.stop_gradient(rc)
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
