"""Test the PSDToACV class."""

import unittest

import sys
sys.path.append('../../src')

import jax.numpy as jnp
from pioran.psdtoacv import PSDToACV
from pioran.psd import Lorentzian,Gaussian,OneBendPowerLaw

T = 1303
dt = 0.0253
S_low = 20
S_high = 10

class TestPSDToACV(unittest.TestCase):
    
    def test_init_FFT(self):
        psd = Lorentzian([1.2,4,5])
        acv = PSDToACV(psd,T=T,dt=dt,S_low=S_low,S_high=S_high,method='FFT')
        self.assertEqual(acv.PSD,psd)

    
    # def test_init_NuFFT(self):
    #     psd = Lorentzian([1.2,4,5])
    #     acv = PSDToACV(psd,T=T,dt=dt,S_low=S_low,S_high=S_high,method='NuFFT')
    #     self.assertEqual(acv.PSD,psd)
    

if __name__ == '__main__':
    unittest.main()
