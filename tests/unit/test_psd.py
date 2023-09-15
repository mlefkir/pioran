"""Tests for the power spectral density function modules."""

import unittest


import unittest
import sys
sys.path.append('../../src')

import jax.numpy as jnp
from pioran.psd_base import PowerSpectralDensity
from pioran.psd import Lorentzian,Gaussian,OneBendPowerLaw

def build_object():
    """Build a power spectral density function object."""
    class mypsd(PowerSpectralDensity):
        expression = 'p1*x+p2'
        def __init__(self, param_values, **kwargs):
            free_parameters = kwargs.get('free_parameters', [True, True])
            PowerSpectralDensity.__init__(self, param_values=param_values,param_names=['p1', 'p2'], free_parameters=free_parameters)
        def calculate(self,x):
            return self.parameters['p1'].value*x+self.parameters['p2'].value
    psd = mypsd([3.1,12.3])
    return psd


class TestPowerSpectrum(unittest.TestCase):
    
    def setUp(self):
        self.lorentz = Lorentzian([1.2,4,5],free_parameters=[True,False,True])
        self.gauss = Gaussian([1.233,3.2432,4.5])
        self.obpl = OneBendPowerLaw([1,3.2432,4.5,6.7])
        self.manualpsd = build_object()
        
    def test_setup_custom_init(self):
        psd = self.manualpsd
        self.assertEqual(psd.parameters[1].value,3.1)
        self.assertEqual(psd.parameters[2].value,12.3)
        self.assertEqual(psd.parameters[1].free,True)
        self.assertEqual(psd.parameters[2].free,True)
    
    def test_custom_calculate(self):
        psd = self.manualpsd
        x = jnp.linspace(0,10,100)
        if not jnp.isclose(psd.calculate(x),3.1*x+12.3,atol=1e-15,rtol=1e-15).all():
            raise AssertionError('The power spectrum is not calculated correctly')
        
    def test_values_params(self):
        self.assertEqual(self.lorentz.parameters[1].value,1.2)
        self.assertEqual(self.lorentz.parameters[2].value,4)
        self.assertEqual(self.lorentz.parameters[3].value,5)
        self.assertEqual(self.lorentz.parameters[1].free,True)
        self.assertEqual(self.lorentz.parameters[2].free,False)
        self.assertEqual(self.lorentz.parameters[3].free,True)
        
        self.assertEqual(self.gauss.parameters[1].value,1.233)
        self.assertEqual(self.gauss.parameters[2].value,3.2432)
        self.assertEqual(self.gauss.parameters[3].value,4.5)
        self.assertEqual(self.gauss.parameters[1].free,True)
        self.assertEqual(self.gauss.parameters[2].free,True)
        self.assertEqual(self.gauss.parameters[3].free,True)
    
    def test_powerspectrum_add_init(self):
        
        model = self.lorentz + self.gauss
        
        self.assertEqual(model.parameters[1].value,1.2)
        self.assertEqual(model.parameters[2].value,4)
        self.assertEqual(model.parameters[3].value,5)
        self.assertEqual(model.parameters[4].value,1.233)
        self.assertEqual(model.parameters[5].value,3.2432)
        self.assertEqual(model.parameters[6].value,4.5)
        
        self.assertEqual(model.parameters[1].free,True)
        self.assertEqual(model.parameters[2].free,False)
        self.assertEqual(model.parameters[3].free,True)
        self.assertEqual(model.parameters[4].free,True)
        self.assertEqual(model.parameters[5].free,True)
        self.assertEqual(model.parameters[6].free,True)
        
    def test_powerspectrum_mul_init(self):
        
        model = self.lorentz * self.gauss
        
        self.assertEqual(model.parameters[1].value,1.2)
        self.assertEqual(model.parameters[2].value,4)
        self.assertEqual(model.parameters[3].value,5)
        self.assertEqual(model.parameters[4].value,1.233)
        self.assertEqual(model.parameters[5].value,3.2432)
        self.assertEqual(model.parameters[6].value,4.5)
        
        self.assertEqual(model.parameters[1].free,True)
        self.assertEqual(model.parameters[2].free,False)
        self.assertEqual(model.parameters[3].free,True)
        self.assertEqual(model.parameters[4].free,True)
        self.assertEqual(model.parameters[5].free,True)
        self.assertEqual(model.parameters[6].free,True)
        
    def test_powerspectrum_add_calculate(self):
            
        model = self.lorentz + self.gauss
        x = jnp.linspace(0,100,1000)
        if not jnp.isclose(model.calculate(x),self.lorentz.calculate(x)+self.gauss.calculate(x),atol=1e-15,rtol=1e-15).all():
            raise AssertionError('The total power spectrum is not calculated correctly')
        
    def test_powerspectrum_mul_calculate(self):
            
        model = self.lorentz * self.gauss
        x = jnp.linspace(0,100,1000)
        if not jnp.isclose(model.calculate(x),self.lorentz.calculate(x)*self.gauss.calculate(x),atol=1e-15,rtol=1e-15).all():
            raise AssertionError('The total power spectrum is not calculated correctly')
        
        
    
    
if __name__ == '__main__':
    unittest.main()
