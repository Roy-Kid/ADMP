import pytest
from openmm import app

class TestForceField:
    
    def test_init(self):
        
        ff = app.ForceField('mpidwater.xml')