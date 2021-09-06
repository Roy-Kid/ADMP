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

class NeighborList:
    
    def __init__(self, positions, box, rc) -> None:
        self.nbs, self.distances2, dr_vecs = NeighborList.construct_nblist(positions, box, rc)
    
    @staticmethod
    def construct_nblist(positions, box, rc):
        
        box_inv = np.linalg.inv(box)
        len_abc_star = np.sqrt(np.sum(box_inv*box_inv, axis=0))
        h_box = 1 / len_abc_star # the "heights" of the box
        n_cells = np.floor(h_box/rc).astype(np.int32)
        n_totc = np.prod(n_cells)
        n_atoms = len(positions)


        cell = np.diag(1./n_cells).dot(box)
        cell_inv = np.linalg.inv(cell)

        upositions = positions.dot(cell_inv)
        indices = np.floor(upositions).astype(np.int32)
        # for safty, do pbc shift
        indices[:, 0] = indices[:, 0] % n_cells[0]
        indices[:, 1] = indices[:, 1] % n_cells[1]
        indices[:, 2] = indices[:, 2] % n_cells[2]
        na, nb, nc = n_cells
        cell_list = LinkedList((na, nb, nc))

        for i_atom in range(n_atoms):
            ia, ib, ic = indices[i_atom]
            cell_list[ia, ib, ic].append(i_atom)


        
        rc2 = rc * rc
        nbs = LinkedList((n_atoms, ))
        distances2 = LinkedList((n_atoms, ))
        dr_vecs = LinkedList((n_atoms, ))

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
                                        ri = positions[catom]
                                        rj = positions[natom]
                                        dr = ri - rj
                                        ds = np.dot(dr, box_inv)
                                        ds -= np.floor(ds+0.5)
                                        dr = np.dot(ds, box)
                                        drdr = dr@dr
                                        if drdr < rc2:
                                            if natom not in nbs[catom] and natom != catom:
                                                nbs[catom].append(natom)
                                                distances2[catom].append(drdr)
                                                dr_vecs[catom].append(dr)
                                            if catom not in nbs[natom] and catom != natom:
                                                nbs[natom].append(catom)
                                                distances2[natom].append(drdr)
                                                dr_vecs[natom].append(-dr)
        return nbs, distances2, dr_vecs
                                            
