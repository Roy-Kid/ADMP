
from admp.forcefield import ForceField

class TestForceField:
    
    def test_load_file(self):
        
        ff = ForceField(f'mpidwater.xml')
        
        