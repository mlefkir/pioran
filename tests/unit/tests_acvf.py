"""Tests for the autocovariance function modules."""

import unittest
import sys
sys.path.append('../../src')

import jax.numpy as jnp
from pioran.acvf_base import CovarianceFunction
from pioran.acvf import Exponential,SquaredExponential

def build_object():
    """Build a covariance function object."""
    class mycov(CovarianceFunction):
        expression = 'p1*x+p2'
        def __init__(self, param_values, **kwargs):
            free_parameters = kwargs.get('free_parameters', [True, True])
            CovarianceFunction.__init__(self, param_values=param_values,param_names=['p1', 'p2'], free_parameters=free_parameters)
        def calculate(self,x):
            return self.parameters['p1'].value*x+self.parameters['p2'].value
    cov = mycov([4.1,3.52])
    return cov

class TestCovarianceFunction(unittest.TestCase):
    
    def setUp(self) :
        self.expocov = Exponential([1.2,4],free_parameters=[True,False])
        self.sqrtcov = SquaredExponential([3.2432,4.5])
        self.manualcov = build_object()

    def test_setup_custom_init(self):
        cov = self.manualcov
        self.assertEqual(cov.parameters[1].value,4.1)
        self.assertEqual(cov.parameters[2].value,3.52)
               
    def test_setup_custom_calculate(self):

        cov = self.manualcov
        x = jnp.linspace(0,10,100)
        if not jnp.isclose(cov.calculate(x),4.1*x+3.52,atol=1e-15,rtol=1e-15).all():
            raise AssertionError('The covariance function is not calculated correctly')

    def test_covariancefunction_init(self):
        
        self.assertEqual(self.expocov.parameters[1].value,1.2)
        self.assertEqual(self.expocov.parameters[2].value,4)
        self.assertEqual(self.expocov.parameters[1].free,True)
        self.assertEqual(self.expocov.parameters[2].free,False)
        
        self.assertEqual(self.sqrtcov.parameters[1].value,3.2432)
        self.assertEqual(self.sqrtcov.parameters[2].value,4.5)
        self.assertEqual(self.sqrtcov.parameters[1].free,True)
        self.assertEqual(self.sqrtcov.parameters[2].free,True)
        
    def test_covariancefunction_add_init(self):
        """Test the addition of covariance functions.
        """
        model = self.expocov + self.sqrtcov

        self.assertEqual(model.parameters[1].value,1.2)
        self.assertEqual(model.parameters[2].value,4)
        self.assertEqual(model.parameters[1].free,True)
        self.assertEqual(model.parameters[2].free,False)
        
        self.assertEqual(model.parameters[3].value,3.2432)
        self.assertEqual(model.parameters[4].value,4.5)
        self.assertEqual(model.parameters[3].free,True)
        self.assertEqual(model.parameters[4].free,True)
        
        self.assertEqual(model.parameters.IDs,[1,2,3,4])
        
    def test_covariancefunction_mul_init(self):
        """Test the multiplication of covariance functions.
        """
        model = self.expocov * self.sqrtcov
        
        self.assertEqual(model.parameters[1].value,1.2)
        self.assertEqual(model.parameters[2].value,4)
        self.assertEqual(model.parameters[1].free,True)
        self.assertEqual(model.parameters[2].free,False)
        
        self.assertEqual(model.parameters[3].value,3.2432)
        self.assertEqual(model.parameters[4].value,4.5)
        self.assertEqual(model.parameters[3].free,True)
        self.assertEqual(model.parameters[4].free,True)
        
        self.assertEqual(model.parameters.IDs,[1,2,3,4])

           
    def test_covariancefunction_add_calculate(self):
        """Test the addition of covariance functions.
        """
                
        t = jnp.linspace(0,10,100)
        model = self.expocov + self.sqrtcov
        
        if not jnp.isclose(model.calculate(t),self.expocov.calculate(t)+self.sqrtcov.calculate(t),atol=1e-15,rtol=1e-15).all():
            print(model.calculate(t)-(self.expocov.calculate(t)+self.sqrtcov.calculate(t)))
            raise AssertionError('The covariance functions are not added correctly')
    
    def test_covariancefunction_mul_calculate(self):
        """Test the multiplication of covariance functions.
        """
        
        t = jnp.linspace(0,10,100)
        model = self.expocov * self.sqrtcov
        
        if not jnp.isclose(model.calculate(t),self.expocov.calculate(t)*self.sqrtcov.calculate(t),atol=1e-15,rtol=1e-15).all():
            print(model.calculate(t)-(self.expocov.calculate(t)*self.sqrtcov.calculate(t)))
            raise AssertionError('The covariance functions are not added correctly')
                            
    
    
if __name__ == '__main__':
    unittest.main()
