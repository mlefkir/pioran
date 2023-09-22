"""unit tests for core.py module
"""

import unittest

import sys
sys.path.append('../../src')

import jax
import jax.numpy as jnp

from pioran.acvf import Exponential
from pioran.psd import OneBendPowerLaw



class TestPowerSpectrum(unittest.TestCase):
    
    def setUp(self):
        self.expo = Exponential([12.1,0.03])
        self.obpl = OneBendPowerLaw([1,3.2432,4.5,6.7])
        # self.manualpsd = build_object()
        
    def test_setup_custom_init(self):
        
        # psd = self.manualpsd
        # self.assertEqual(psd.parameters[1].value,3.1)
        # self.assertEqual(psd.parameters[2].value,12.3)
        # self.assertEqual(psd.parameters[1].free,True)
        # self.assertEqual(psd.parameters[2].free,True)
    

    
if __name__ == '__main__':
    unittest.main()
