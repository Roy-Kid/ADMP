from functools import partial
import numpy as np
import jax
import jax.numpy as jnp

class LinkedList:
    
    def __init__(self, shape, container=list) -> None:
        
        nele = np.prod(shape)
        self.indices = np.empty(nele)
        self.map = {}
        for i in range(nele):
            k = container()
            self.indices[i] = id(k)
            self.map[id(k)] = k
        self.indices = self.indices.reshape(shape)

    def __getitem__(self, i):

        if isinstance(i, int):
            return self.map[self.indices[i]]
        elif isinstance(i, slice):
            return [self.map[k] for k in self.indices[i]]
        else:
            return self.map[self.indices[tuple(i)]]
        
        
class NBLS:
    '''
    The container object for neighbour list information
    '''
    
    def __init__(self, nbs, distances2, dr_vecs, R):
        self.nbs = nbs
        self.distances2 = distances2
        self.dr_vecs = dr_vecs
        self.R = R

def construct_nblist(positions, box, rc):
    box_inv = jnp.linalg.inv(box)
    len_abc_star = jnp.sqrt(jnp.sum(box_inv*box_inv, axis=0))
    h_box = 1 / len_abc_star # the "heights" of the box
    n_cells = jnp.floor(h_box/rc).astype(np.int32)
    n_totc = np.prod(n_cells)
    n_atoms = len(positions)


    cell = jnp.diag(1./n_cells).dot(box)
    cell_inv = jnp.linalg.inv(cell)

    indices = jnp.floor(positions.dot(cell_inv)).astype(jnp.int32)
    # for safty, do pbc shift
    indices = jnp.mod(indices, n_cells[jnp.newaxis])
    na, nb, nc = n_cells
    cell_list = LinkedList((na, nb, nc))

    for i_atom in range(n_atoms):
        ia, ib, ic = indices[i_atom]
        cell_list[ia, ib, ic].append(i_atom)


    
    rc2 = rc * rc
    nbs = [] # pair-wise
    distances2 = [] # 
    dr_vecs = []
    R = []

    for ic in range(na):
        for jc in range(nb):
            for kc in range(nc):
                center_atoms = cell_list[ic, jc, kc]
                # search neighbor cells
                for i in (-1, 0, 1):
                    for j in (-1, 0, 1):
                        for k in (-1, 0, 1):
                            neighbor_atoms = cell_list[i, j, k]
                            for catom in center_atoms:
                                for natom in neighbor_atoms:
                                    if catom == natom:
                                        continue
                                    else:
                                        pair = sorted([catom, natom])
                                        c, n = pair
                                    assert c < n
                                    ri = positions[c]
                                    rj = positions[n]
                                    dr = rj - ri
                                    ds = jnp.dot(dr, box_inv)
                                    ds -= jnp.floor(ds+0.5)
                                    dr = jnp.dot(ds, box)
                                    drdr = dr@dr
                                    if drdr < rc2 and pair not in nbs:
                                        nbs.append(pair)
                                        distances2.append(drdr)
                                        dr_vecs.append(dr)
                                        
                                        vectorZ = dr/drdr**0.5
                                        vectorX = jnp.array(vectorZ)
                                        if ri[1] != rj[1] or ri[2] != rj[2]:
                                            vectorX = vectorX.at[0].add(1)
                                        else:
                                            vectorX = vectorX.at[1].add(1)
                                        dot = jnp.dot(vectorZ, vectorX)
                                        vectorX -= vectorZ * dot
                                        vectorX = vectorX / jnp.linalg.norm(vectorX)
                                        vectorY = jnp.cross(vectorZ, vectorX)
                                        R.append(jnp.array([vectorX, vectorY, vectorZ]))
  
    
    return NBLS(nbs, distances2, dr_vecs, R)