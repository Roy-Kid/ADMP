from python.ADMPForce import read_mpid_inputs

def approx(value1, value2, prec):
    
    v1 = str(value1)
    v2 = str(value2)

def test_water_dimer(water_dimer):
    generator = water_dimer
    force = generator.create_force()
    force.update()
    force.kappa = 0.328532611
    assert force.kappa == 0.328532611
    assert force.calc_real_space_energy() == 878.8693561091234
        
def test_read_mpid():
    positions, box, list_elem, mpid_params = read_mpid_inputs('/home/roy/work/multi_pme/mtpl_pme/waterdimer_aligned.pdb', '/home/roy/work/multi_pme/mtpl_pme/mpidwater.xml')
