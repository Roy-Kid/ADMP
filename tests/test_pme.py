from python.ADMPForce import read_mpid_inputs
import pytest
@pytest.fixture
def force(water_dimer):
    generator = water_dimer
    force = generator.create_force()
    force.update()
    force.kappa = 0.328532611
    yield force

def approx(value1, value2, prec):
    prec -= 1
    fluc = 10**(-1*prec)
    if (value2 - fluc) < value1 and (value2 + fluc) > value1:
        return True
    else:
        return False

def test_real_space(force):

    # test real_space_energy
    real_e = force.calc_real_space_energy()
    assert real_e == 878.8693561091234
    assert approx(force.kappa, 0.32853, 5)
    assert approx(real_e, 878.869356, 6)
    
def test_self(force):
    self_e = force.calc_self_energy()
    assert approx(self_e, -872.4393, 5)
    
def test_reci_space(force):
    # test reci_space_energy
    reci_e = force.calc_reci_space_energy()
    # assert reci_e == 3.850
    assert approx(reci_e, 3.850, 4)
    
    self_e = force.calc_self_energy()
    assert approx(self_e, -872.4393, 5)
    
        
def test_read_mpid():
    positions, box, list_elem, mpid_params = read_mpid_inputs('/home/roy/work/multi_pme/mtpl_pme/waterdimer_aligned.pdb', '/home/roy/work/multi_pme/mtpl_pme/mpidwater.xml')