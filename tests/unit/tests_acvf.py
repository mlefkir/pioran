

import unittest
import sys
sys.path.append('../../src')

import jax.numpy as jnp
from pioran.acvf_base import CovarianceFunction
from pioran.acvf import Exponential,SquaredExponential


class TestCovarianceFunction(unittest.TestCase):
    
    def test_covariancefunction_init(self):
        expocov = Exponential([1.2,4],free_parameters=[True,False])
        sqrtcov = SquaredExponential([3.2432,4.5])
        
        self.assertEqual(expocov.parameters[1].value,1.2)
        self.assertEqual(expocov.parameters[2].value,4)
        self.assertEqual(expocov.parameters[1].free,True)
        self.assertEqual(expocov.parameters[2].free,False)
        
        self.assertEqual(sqrtcov.parameters[1].value,3.2432)
        self.assertEqual(sqrtcov.parameters[2].value,4.5)
        self.assertEqual(sqrtcov.parameters[1].free,True)
        self.assertEqual(sqrtcov.parameters[2].free,True)
        
    def test_covariancefunction_add_init(self):
        """Test the addition of covariance functions.
        """
        expocov = Exponential([1.2,4],free_parameters=[True,False])
        sqrtcov = SquaredExponential([3.2432,4.5])
        model = expocov + sqrtcov
        
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
        expocov = Exponential([1.2,4],free_parameters=[True,False])
        sqrtcov = SquaredExponential([3.2432,4.5])
        model = expocov * sqrtcov
        
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
        
        expocov = Exponential([1.2,4],free_parameters=[True,False])
        sqrtcov = SquaredExponential([3.2432,4.5])
        t = jnp.linspace(0,10,100)
        model = expocov + sqrtcov
        
        if not jnp.isclose(model.calculate(t),expocov.calculate(t)+sqrtcov.calculate(t),atol=1e-15,rtol=1e-15).all():
            print(model.calculate(t)-(expocov.calculate(t)+sqrtcov.calculate(t)))
            raise AssertionError('The covariance functions are not added correctly')
    
    def test_covariancefunction_mul_calculate(self):
        """Test the multiplication of covariance functions.
        """
        
        expocov = Exponential([1.2,4],free_parameters=[True,False])
        sqrtcov = SquaredExponential([3.2432,4.5])
        t = jnp.linspace(0,10,100)
        model = expocov * sqrtcov
        
        if not jnp.isclose(model.calculate(t),expocov.calculate(t)*sqrtcov.calculate(t),atol=1e-15,rtol=1e-15).all():
            print(model.calculate(t)-(expocov.calculate(t)*sqrtcov.calculate(t)))
            raise AssertionError('The covariance functions are not added correctly')
                            
    
    
if __name__ == '__main__':
    unittest.main()
