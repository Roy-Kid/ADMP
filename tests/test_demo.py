# author: lijichen
#         yanglan
#         chenjunmin
# contact: lijichen365@126.com
# date: 2021-12-21
# version: 0.0.1
from admp.demo import *
import pytest

def test_demo():
    
    pdb = 'tests/samples/water1.pdb'
    #pdb = f'tests/samples/water1.pdb'
    xml = 'tests/samples/mpidwater.xml'

    # return a dict with raw data from pdb
    pdbinfo = read_pdb(pdb)

    serials = pdbinfo['serials']
    names = pdbinfo['names']
    resNames = pdbinfo['resNames']
    resSeqs = pdbinfo['resSeqs']
    positions = pdbinfo['positions']
    charges = pdbinfo['charges']
    box = pdbinfo['box'] # a, b, c, α, β, γ
    positions = jnp.asarray(positions)
    lx = box[0]
    ly = box[1]
    lz = box[2]
    
    dielectric = 1389.35455846 # in e^2/A
    mScales = jnp.array([0.0, 0.0, 0.0, 1.0, 1.0])
    pScales = jnp.array([0.0, 0.0, 0.0, 1.0, 1.0])
    dScales = jnp.array([0.0, 0.0, 0.0, 1.0, 1.0])

    box = jnp.eye(3)*jnp.array([lx, ly, lz])

    rc = 4  # in Angstrom
    ethresh = 1e-4

    natoms = len(serials)

    # Here are the templates[dict] from xml. 
    # atomTemplates are per-atom info,
    # residueTemplates contain what atoms in the residue and how they connect. 
    atomTemplate, residueTemplate = read_xml(xml)

    # then we use the template to init atom and residue objects
    # TODO: an atomManager and residueManager
    atomDicts, residueDicts = init_residues(serials, names, resNames, resSeqs, positions, charges, atomTemplate, residueTemplate)

    Q = np.vstack(
        [(atom.c0, atom.dX*10, atom.dY*10, atom.dZ*10, atom.qXX*300, atom.qYY*300, atom.qZZ*300, atom.qXY*300, atom.qXZ*300, atom.qYZ*300) for atom in atomDicts.values()]
    )

    Qlocal = convert_cart2harm(Q, 2)
    Qlocal = jnp.array(Qlocal)
    axis_type = np.array(
        [atom.axisType for atom in atomDicts.values()]
    )

    axis_indices = np.vstack(
        [atom.axis_indices for atom in atomDicts.values()]
    )
    # covalent_map is simply as N*N matrix now
    # remove sparse matrix
    covalent_map = assemble_covalent(residueDicts, natoms)
    ## TODO: setup kappa
    kappa, K1, K2, K3 = setup_ewald_parameters(rc, ethresh, box)
    kappa = 0.657065221219616 
    # === pre-compile compute function ===
    jitted_pme_reci_energy_and_force = jit(value_and_grad(gen_pme_reciprocal(axis_type, axis_indices)), static_argnums=(4,5,6,7))
    construct_localframes = generate_construct_localframes(axis_type, axis_indices)
    jitted_construct_localframes = jit(construct_localframes)
    jitted_pme_real_energy_and_force = jit(value_and_grad(real_space, argnums=0))

    jitted_disp_real_energy_and_force = jit(value_and_grad(dispersion_real))
    jitted_disp_self_energy_and_force = jit(value_and_grad(dispersion_self))
    jitted_disp_reci_energy_and_force = jit(value_and_grad(dispersion_reciprocal),static_argnums=(4,5,6))
    # === prepare neighborlist ===
    
    cell_size = rc
    displacement_fn, shift_fn = space.periodic_general(box, fractional_coordinates=False)
    neighbor_list_fn = partition.neighbor_list(displacement_fn, box, cell_size, 0)
    start_nbl = time.time()
    nbr = neighbor_list_fn.allocate(positions)
    #nbr = nbr.update(positions)


    # == C_list
    C_list = np.random.rand(3,natoms)
    nmol=int(natoms/3)
    for i in range(nmol):
        a = i*3
        b = i*3+1
        c = i*3+2
        C_list[0][a]=37.19677405
        C_list[0][b]=7.6111103
        C_list[0][c]=7.6111103
        C_list[1][a]=85.26810658
        C_list[1][b]=11.90220148
        C_list[1][c]=11.90220148
        C_list[2][a]=134.44874488
        C_list[2][b]=15.05074749
        C_list[2][c]=15.05074749 
    end_nbl = time.time()

    # === start to calculate ADMP === 
    ereal, freal = jitted_pme_real_energy_and_force(positions, Qlocal, box, kappa)
    ereci, freci = jitted_pme_reci_energy_and_force(positions, box, Qlocal, kappa, 2, K1, K2, K3)
    eself, fself = jitted_pme_self_energy_and_force(Qlocal, kappa)

    edisp_real, fdisp_real = jitted_disp_real_energy_and_force(positions,C_list,kappa,mScales)
    edisp_self, fdisp_self = jitted_disp_self_energy_and_force(C_list, kappa)    
    edisp_reci, fdisp_reci = jitted_disp_reci_energy_and_force(positions, box, C_list, kappa, K1, K2, K3)
    # --- start calc pme ---
    epoch = 100
    
    #start_real = time.time()
    #for i in range(epoch):
        #ereal, freal = jitted_pme_real_energy_and_force(positions, Qlocal, box, kappa)
        #ereal.block_until_ready()
        #freal.block_until_ready()
    #end_real = time.time()

    #start_reci = time.time()
    #for i in range(epoch):
        #ereci, freci = jitted_pme_reci_energy_and_force(positions, box, Qlocal, kappa, 2, K1, K2, K3)
    #ereci.block_until_ready()
    #freci.block_until_ready()
    #end_reci = time.time()
    
    #start_self = time.time()
    #for i in range(epoch):
        #eself, fself = jitted_pme_self_energy_and_force(Qlocal, kappa)
    #eself.block_until_ready()
    #self.block_until_ready()
    #end_self = time.time()
    
    start_real = time.time()
    for i in range(epoch):
        edisp_real, fdisp_real = jitted_disp_real_energy_and_force(positions,C_list,kappa,mScales)
    edisp_real.block_until_ready()
    fdisp_real.block_until_ready()
    end_real = time.time()

    start_self = time.time()
    for i in range(epoch):
        edisp_self, fdisp_self = jitted_disp_self_energy_and_force(C_list, kappa) 
    edisp_self.block_until_ready()
    fdisp_self.block_until_ready()
    end_self = time.time()

    start_reci = time.time()
    for i in range(epoch):
        edisp_reci ,fdisp_reci = jitted_disp_reci_energy_and_force(positions, box, C_list, kappa, K1, K2, K3)
    edisp_reci.block_until_ready()
    fdisp_reci.block_until_ready()
    end_reci = time.time()
    print(ereal)
    print(ereci)
    print(eself)
    print(f'total : {edisp_real+edisp_reci+edisp_self}')
    print(edisp_real)
    print(edisp_reci)
    print(edisp_self)
    print(f'real e&f cost: {(end_real - start_real)/epoch}')
    print(f'reci e&f cost: {(end_reci - start_reci)/epoch}')
    print(f'self e&f cost: {(end_self - start_self)/epoch}')  
