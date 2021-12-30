import pytest
import numpy as np
import jax.numpy as jnp
from admp.spatial import pbc_shift, v_pbc_shift, normalize
import numpy.testing as npt

class TestPBC:
    
    @pytest.mark.parametrize('drvecs, expected_rvecs', [
        (jnp.array([[0, 0, 0]]), jnp.array([[0, 0, 0]])), 
        (jnp.array([[5, 5, 5]]), jnp.array([[-5, -5, -5]])), 
        (jnp.array([[10, 10, 10], [15, 0, 0]]), jnp.array([[0, 0, 0], [-5, 0, 0]]))
    ])
    def test_pbc_shift(self, drvecs, expected_rvecs):
        box = jnp.array([[10, 0, 0], [0, 10, 0], [0, 0, 10]])
        box_inv = jnp.linalg.inv(box)
        
        rvecs = pbc_shift(drvecs, box, box_inv)
        npt.assert_allclose(rvecs, expected_rvecs)
        rvecs = v_pbc_shift(drvecs, box, box_inv)
        npt.assert_allclose(rvecs, expected_rvecs)
