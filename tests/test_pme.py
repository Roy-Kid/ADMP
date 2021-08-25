from python.ADMPForce import read_mpid_inputs

def approx(value1, value2, prec):
    prec -= 1
    fluc = 10**(-1*prec)
    if (value2 - fluc) < value1 and (value2 + fluc) > value1:
        return True
    else:
        return False

def test_water_dimer(water_dimer):
    generator = water_dimer
    force = generator.create_force()
    force.update()
    force.kappa = 0.328532611
    assert force.kappa == 0.328532611
    real_e = force.calc_real_space_energy()
    assert real_e == 878.8693561091234
    assert approx(force.kappa, 0.32853, 5)
    assert approx(real_e, 878.869356, 6)
        
def test_read_mpid():
    positions, box, list_elem, mpid_params = read_mpid_inputs('/home/roy/work/multi_pme/mtpl_pme/waterdimer_aligned.pdb', '/home/roy/work/multi_pme/mtpl_pme/mpidwater.xml')
