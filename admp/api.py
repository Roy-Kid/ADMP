#!/usr/bin/env python
import sys
import openmm
import openmm.app as app
import openmm.unit as unit
import numpy as np
import jax.numpy as jnp
from collections import defaultdict
from admp.disp_pme import ADMPDispPmeForce
from admp.multipole import convert_cart2harm, rot_local2global
from admp.pairwise import TT_damping_qq_c6_kernel, generate_pairwise_interaction
from admp.pme import ADMPPmeForce
from admp.spatial import generate_construct_local_frames
from admp.recip import Ck_1, generate_pme_recip
from jax_md import space, partition
from jax import grad
import linecache


def get_line_context(file_path, line_number):
    return linecache.getline(file_path,line_number).strip()


def build_covalent_map(data, max_neighbor):
    n_atoms = len(data.atoms)
    covalent_map = np.zeros((n_atoms, n_atoms), dtype=int)
    for bond in data.bonds:
        covalent_map[bond.atom1, bond.atom2] = 1
        covalent_map[bond.atom2, bond.atom1] = 1
    for n_curr in range(1, max_neighbor):
        for i in range(n_atoms):
            # current neighbors
            j_list = np.where(
                np.logical_and(covalent_map[i] <= n_curr, covalent_map[i] > 0)
            )[0]
            for j in j_list:
                k_list = np.where(covalent_map[j] == 1)[0]
                for k in k_list:
                    if k != i and k not in j_list:
                        covalent_map[i, k] = n_curr + 1
                        covalent_map[k, i] = n_curr + 1
    return covalent_map


def set_axis_type(map_atomtypes, types, params):

    ZThenX = 0
    Bisector = 1
    ZBisect = 2
    ThreeFold = 3
    Zonly = 4
    NoAxisType = 5
    LastAxisTypeIndex = 6
    kStrings = ["kz", "kx", "ky"]
    axisIndices = []
    axisTypes = []

    for i in map_atomtypes:
        atomType = types[i]

        kIndices = [atomType]

        for kString in kStrings:
            kString_value = params[kString][i]
            if kString_value != "":
                kIndices.append(kString_value)
        axisIndices.append(kIndices)

        # set axis type

        kIndicesLen = len(kIndices)

        if kIndicesLen > 3:
            ky = kIndices[3]
            kyNegative = False
            if ky.startswith("-"):
                ky = kIndices[3] = ky[1:]
                kyNegative = True
        else:
            ky = ""

        if kIndicesLen > 2:
            kx = kIndices[2]
            kxNegative = False
            if kx.startswith("-"):
                kx = kIndices[2] = kx[1:]
                kxNegative = True
        else:
            kx = ""

        if kIndicesLen > 1:
            kz = kIndices[1]
            kzNegative = False
            if kz.startswith("-"):
                kz = kIndices[1] = kz[1:]
                kzNegative = True
        else:
            kz = ""

        while len(kIndices) < 4:
            kIndices.append("")

        axisType = ZThenX
        if not kz:
            axisType = NoAxisType
        if kz and not kx:
            axisType = Zonly
        if kz and kzNegative or kx and kxNegative:
            axisType = Bisector
        if kx and kxNegative and ky and kyNegative:
            axisType = ZBisect
        if kz and kzNegative and kx and kxNegative and ky and kyNegative:
            axisType = ThreeFold

        axisTypes.append(axisType)

    return np.array(axisTypes), np.array(axisIndices)


class ADMPDispGenerator:
    def __init__(self, hamiltonian):
        self.ff = hamiltonian
        self.params = {"A": [], "B": [], "Q": [], "C6": [], "C8": [], "C10": []}
        self._jaxPotential = None
        self.types = []
        self.ethresh = 1.0e-5
        self.pmax = 10

    def registerAtomType(self, atom):
        self.types.append(atom["type"])
        self.params["A"].append(float(atom["A"]))
        self.params["B"].append(float(atom["B"]))
        self.params["Q"].append(float(atom["Q"]))
        self.params["C6"].append(float(atom["C6"]))
        self.params["C8"].append(float(atom["C8"]))
        self.params["C10"].append(float(atom["C10"]))

    @staticmethod
    def parseElement(element, hamiltonian):
        generator = ADMPDispGenerator(hamiltonian)
        hamiltonian.registerGenerator(generator)
        # covalent scales
        mScales = []
        for i in range(2, 7):
            mScales.append(float(element.attrib["mScale1%d" % i]))
        generator.params["mScales"] = mScales
        for atomtype in element.findall("Atom"):
            generator.registerAtomType(atomtype.attrib)
        # jax it!
        for k in generator.params.keys():
            generator.params[k] = jnp.array(generator.params[k])
        generator.types = np.array(generator.types)

    def createForce(self, system, data, nonbondedMethod, nonbondedCutoff, args):

        n_atoms = len(data.atoms)
        # build index map
        map_atomtype = np.zeros(n_atoms, dtype=int)
        for i in range(n_atoms):
            atype = data.atomType[data.atoms[i]]
            map_atomtype[i] = np.where(self.types == atype)[0][0]
        # build covalent map
        covalent_map = build_covalent_map(data, 6)

        # here box is only used to setup ewald parameters, no need to be differentiable
        a, b, c = system.getDefaultPeriodicBoxVectors()
        box = jnp.array([a._value, b._value, c._value]) * 10
        # get the admp calculator
        rc = nonbondedCutoff.value_in_unit(unit.angstrom)

        # get calculator
        Force_DispPME = ADMPDispPmeForce(box, covalent_map, rc, self.ethresh, self.pmax)
        # debugging
        # Force_DispPME.update_env('kappa', 0.657065221219616)
        # Force_DispPME.update_env('K1', 96)
        # Force_DispPME.update_env('K2', 96)
        # Force_DispPME.update_env('K3', 96)
        pot_fn_lr = Force_DispPME.get_energy
        pot_fn_sr = generate_pairwise_interaction(
            TT_damping_qq_c6_kernel, covalent_map, static_args={}
        )

        def potential_fn(positions, box, pairs, params):
            mScales = params["mScales"]
            a_list = (
                params["A"][map_atomtype] / 2625.5
            )  # kj/mol to au, as expected by TT_damping kernel
            b_list = params["B"][map_atomtype] * 0.0529177249  # nm^-1 to au
            q_list = params["Q"][map_atomtype]
            c6_list = jnp.sqrt(params["C6"][map_atomtype] * 1e6)
            c8_list = jnp.sqrt(params["C8"][map_atomtype] * 1e8)
            c10_list = jnp.sqrt(params["C10"][map_atomtype] * 1e10)
            c_list = jnp.vstack((c6_list, c8_list, c10_list))

            E_sr = pot_fn_sr(
                positions, box, pairs, mScales, a_list, b_list, q_list, c_list[0]
            )
            E_lr = pot_fn_lr(positions, box, pairs, c_list.T, mScales)
            return E_sr - E_lr

        self._jaxPotential = potential_fn
        # self._top_data = data

    def getJaxPotential(self):
        return self._jaxPotential

    def renderXML(self):
        # generate xml force field file
        pass


# register all parsers
app.forcefield.parsers["ADMPDispForce"] = ADMPDispGenerator.parseElement


class ADMPPmeGenerator:
    def __init__(self, hamiltonian):
        self.ff = hamiltonian
        self.kStrings = {
            "kz": [],
            "kx": [],
            "ky": [],
        }
        self._input_params = {
            "c0": [],
            "dX": [],
            "dY": [],
            "dZ": [],
            "qXX": [],
            "qXY": [],
            "qYY": [],
            "qXZ": [],
            "qYZ": [],
            "qZZ": [],
            "oXXX": [],
            "oXXY": [],
            "oXYY": [],
            "oYYY": [],
            "oXXZ": [],
            "oXYZ": [],
            "oYYZ": [],
            "oXZZ": [],
            "oYZZ": [],
            "oZZZ": [],
            "thole": [],
            "polarizabilityXX": [],
            "polarizabilityYY": [],
            "polarizabilityZZ": []
        }
        # if more or optional input params
        # self._input_params = defaultDict(list) 
        self._jaxPotential = None
        self.types = []
        self.ethresh = 1.0e-5
        self.params = {}
        self.lpol = False
        self.ref_dip = ''

    def registerAtomType(self, atom:dict):
        
        self.types.append(atom.pop("type"))

        kStrings = ["kz", "kx", "ky"]
        for kString in kStrings:
            if kString in atom:
                self.kStrings[kString].append(atom.pop(kString))
            else:
                self.kStrings[kString].append("")

        for k, v in atom.items():
            self._input_params[k].append(float(v))

    @staticmethod
    def parseElement(element, hamiltonian):
        generator = ADMPPmeGenerator(hamiltonian)
        generator.lmax = int(element.attrib.get('lmax'))
        generator.pmax = int(element.attrib.get('pmax'))
        
        hamiltonian.registerGenerator(generator)

        mScales = []
        pScales = []
        dScales = []
        for i in range(2, 7):
            mScales.append(float(element.attrib["mScale1%d" % i]))
            pScales.append(float(element.attrib["pScale1%d" % i]))
            dScales.append(float(element.attrib["dScale1%d" % i]))
        generator.params["mScales"] = jnp.array(mScales)
        generator.params["pScales"] = jnp.array(pScales)
        generator.params["dScales"] = jnp.array(dScales)
        
        if element.findall('Polarize'):
            generator.lpol = True

        for atomType in element.findall("Atom"):
            atomAttrib = atomType.attrib
            for polarInfo in element.findall("Polarize"):
                polarAttrib = polarInfo.attrib
                if polarInfo.attrib['type'] == atomAttrib['type']:
                    atomAttrib.update(polarAttrib)
                    break
            generator.registerAtomType(atomAttrib)

        for k in generator._input_params.keys():
            generator._input_params[k] = jnp.array(generator._input_params[k])
        generator.types = np.array(generator.types)

    def createForce(self, system, data, nonbondedMethod, nonbondedCutoff, args):

        n_atoms = len(data.atoms)
        # build index map
        map_atomtype = np.zeros(n_atoms, dtype=int)
        for i in range(n_atoms):
            atype = data.atomType[data.atoms[i]]
            map_atomtype[i] = np.where(self.types == atype)[0][0]
        
        # map atom multipole moments
        p = self._input_params
        Q = np.zeros((n_atoms, 10))
        Q[:, 0] = p["c0"][map_atomtype]
        Q[:, 1] = p["dX"][map_atomtype] * 10
        Q[:, 2] = p["dY"][map_atomtype] * 10
        Q[:, 3] = p["dZ"][map_atomtype] * 10
        Q[:, 4] = p["qXX"][map_atomtype] * 300
        Q[:, 5] = p["qYY"][map_atomtype] * 300
        Q[:, 6] = p["qZZ"][map_atomtype] * 300
        Q[:, 7] = p["qXY"][map_atomtype] * 300
        Q[:, 8] = p["qXZ"][map_atomtype] * 300
        Q[:, 9] = p["qYZ"][map_atomtype] * 300
        
        # map polarization-related params
        pol = jnp.vstack((p['polarizabilityXX'][map_atomtype], p['polarizabilityYY'][map_atomtype], p['polarizabilityZZ'][map_atomtype])).T.astype(jnp.float32)
        pol = 1000*jnp.mean(pol,axis=1)
        self.params['pol'] = pol
        
        tholes = jnp.array(p['thole'][map_atomtype]).astype(jnp.float32)
        tholes = jnp.mean(jnp.atleast_2d(tholes), axis=1)
        self.params['tholes'] = tholes
        
        # defaultTholeWidth = 8
        Uind_global = jnp.zeros([n_atoms,3])
        ref_dip = self.ref_dip
        for i in range(n_atoms):
            a = get_line_context(ref_dip, i+1)
            b = a.split()
            t = np.array([10*float(b[0]),10*float(b[1]),10*float(b[2])])
            Uind_global = Uind_global.at[i].set(t)
            
        # construct the C list
        c_list = np.zeros((3, n_atoms))
        a_list = np.zeros(n_atoms)
        q_list = np.zeros(n_atoms)
        b_list = np.zeros(n_atoms)
        
        
        nmol=int(n_atoms/3) # WARNING: HARD CODE!
        for i in range(nmol):
            a = i*3
            b = i*3+1
            c = i*3+2
            # dispersion coeff
            c_list[0][a]=37.19677405
            c_list[0][b]=7.6111103
            c_list[0][c]=7.6111103
            c_list[1][a]=85.26810658
            c_list[1][b]=11.90220148
            c_list[1][c]=11.90220148
            c_list[2][a]=134.44874488
            c_list[2][b]=15.05074749
            c_list[2][c]=15.05074749
            # q
            q_list[a] = -0.741706
            q_list[b] = 0.370853
            q_list[c] = 0.370853
            # b, Bohr^-1
            b_list[a] = 2.00095977
            b_list[b] = 1.999519942
            b_list[c] = 1.999519942
            # a, Hartree
            a_list[a] = 458.3777
            a_list[b] = 0.0317
            a_list[c] = 0.0317
        
        # add all differentiable params to self.params
        Q = jnp.array(Q)
        Q_local = convert_cart2harm(Q, 2)
        self.params["Q_local"] = Q_local
        # here box is only used to setup ewald parameters, no need to be differentiable
        a, b, c = system.getDefaultPeriodicBoxVectors()
        box = jnp.array([a._value, b._value, c._value]) * 10

        # get the admp calculator
        rc = nonbondedCutoff.value_in_unit(unit.angstrom)

        # build covalent map
        covalent_map = build_covalent_map(data, 6)

        # build intra-molecule axis
        self.axis_types, self.axis_indices = set_axis_type(
            map_atomtype, self.types, self.kStrings
        )
        map_axis_indices = []
        # map axis_indices
        for i in range(n_atoms):
            catom = data.atoms[i]
            residue = catom.residue._atoms
            atom_indices = [
                index if index != "" else -1 for index in self.axis_indices[i][1:]
            ]
            for atom in residue:
                if atom == catom:
                    continue
                for i in range(len(atom_indices)):
                    if atom_indices[i] == data.atomType[atom]:
                        atom_indices[i] = atom.index
                        break
            map_axis_indices.append(atom_indices)

        self.axis_indices = np.array(map_axis_indices)
        
        # Finish data preparation
        # -------------------------------------------------------------------------------------
        # parameters should be ready: 
        # geometric variables: positions, box
        # atomic parameters: Q_local, c_list
        # topological parameters: covalent_map, mScales, pScales, dScales
        # general force field setting parameters: rc, ethresh, lmax, pmax
        
        pme_force = ADMPPmeForce(
            box,
            self.axis_types,
            self.axis_indices,
            covalent_map,
            rc,
            self.ethresh,
            self.lmax,
            self.lpol
        )
        self.params['U_ind'] = pme_force.U_ind


        def potential_fn(positions, box, pairs, params):

            mScales = params["mScales"]
            Q_local = params["Q_local"]
            

            # positions, box, pairs, Q_local, mScales
            if self.lpol:
                pScales = params["pScales"]
                dScales = params["dScales"]
                U_ind = params["U_ind"]
                return pme_force.get_energy(positions, box, pairs, Q_local, pol, tholes, mScales, pScales, dScales, U_init=U_ind)
            else: 
                return pme_force.get_energy(positions, box, pairs, Q_local, mScales)

        self._jaxPotential = potential_fn

    def getJaxPotential(self):
        return self._jaxPotential

    def renderXML(self):
        pass


app.forcefield.parsers["ADMPPmeForce"] = ADMPPmeGenerator.parseElement


class Hamiltonian(app.forcefield.ForceField):
    def __init__(self, xmlname):
        super().__init__(xmlname)
        self._potentials = []

    def createPotential(
        self,
        topology,
        nonbondedMethod=app.NoCutoff,
        nonbondedCutoff=1.0 * unit.nanometer,
    ):
        system = self.createSystem(
            topology, nonbondedMethod=nonbondedMethod, nonbondedCutoff=nonbondedCutoff
        )
        # load_constraints_from_system_if_needed
        # create potentials
        for generator in self._forces:
            potentialImpl = generator.getJaxPotential()
            self._potentials.append(potentialImpl)
        return [p for p in self._potentials]


if __name__ == "__main__":
    H = Hamiltonian("forcefield.xml")
    generator = H.getGenerators()[0]
    app.Topology.loadBondDefinitions("residues.xml")
    pdb = app.PDBFile("../water1024.pdb")
    rc = 4.0
    potentials = H.createPotential(pdb.topology, nonbondedCutoff=rc * unit.angstrom)
    pot_disp = potentials[0]

    positions = jnp.array(pdb.positions._value) * 10
    a, b, c = pdb.topology.getPeriodicBoxVectors()
    box = jnp.array([a._value, b._value, c._value]) * 10

    # neighbor list
    displacement_fn, shift_fn = space.periodic_general(
        box, fractional_coordinates=False
    )
    neighbor_list_fn = partition.neighbor_list(
        displacement_fn, box, rc, 0, format=partition.OrderedSparse
    )
    nbr = neighbor_list_fn.allocate(positions)
    pairs = nbr.idx.T

    param_grad = grad(pot_disp, argnums=3)(positions, box, pairs, generator.params)
    print(param_grad)
