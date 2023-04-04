

import unittest
import sys
sys.path.append('../../src')

from pioran import ParametersModel
from pioran import Parameter


class TestParameterIntegration(unittest.TestCase):
    def test_parameters_attributes(self):
        """
        Test the initialization of the ParametersModel class
        """
        pars = ParametersModel(['a', 'b', 'c'], [1, .2, 4.53],[True, False, True],hyperparameters=[True, False, True],components=[1, 2, 1])
        self.assertEqual(type(pars['a']), Parameter)        
        self.assertEqual(type(pars['b']), Parameter)        
        self.assertEqual(type(pars['c']), Parameter)        

        self.assertEqual(pars['a'].name, 'a')
        self.assertEqual(pars['b'].name, 'b')
        self.assertEqual(pars['c'].name, 'c')
        
        self.assertEqual(pars['a'].value, 1)
        self.assertEqual(pars['b'].value, .2)
        self.assertEqual(pars['c'].value, 4.53)
        
        self.assertEqual(pars['a'].free, True)
        self.assertEqual(pars['b'].free, False)
        self.assertEqual(pars['c'].free, True)
        
        self.assertEqual(pars['a'].ID, 1)
        self.assertEqual(pars['b'].ID, 2)
        self.assertEqual(pars['c'].ID, 3)
        
        self.assertEqual(pars['a'].hyperparameter, True)
        self.assertEqual(pars['b'].hyperparameter, False)
        self.assertEqual(pars['c'].hyperparameter, True)
        
        self.assertEqual(pars['a'].component, 1)
        self.assertEqual(pars['b'].component, 2)
        self.assertEqual(pars['c'].component, 1)
        
    def test_parameters_pars(self):
        """
        """
        pars = ParametersModel(['a', 'b', 'c'], [1, .2, 4.53],[True, False, True])
        self.assertEqual(pars['a'], pars._pars[0])
        self.assertEqual(pars['b'], pars._pars[1])
        self.assertEqual(pars['c'], pars._pars[2])
    
    def test_parameters_setfreevalue(self):
        pars = ParametersModel(['a', 'b', 'c'], [1, .2, .53],[True, False, True])
        pars.set_free_values([42.4, 3])
        self.assertEqual(pars['a'].value, 42.4)
        self.assertEqual(pars['b'].value, .2)
        self.assertEqual(pars['c'].value, 3)
        
        self.assertEqual(pars.free_values, [42.4, 3])
        
    
    def tests_parameters_setnames(self):
        pars = ParametersModel(['a', 'b', 'c'], [1, .2, .53],[True, False, True])
        pars.set_names(['d', 'e', 'f'])
        self.assertEqual(pars['d'].name, 'd')
        self.assertEqual(pars['e'].name, 'e')
        self.assertEqual(pars['f'].name, 'f')
        
        self.assertEqual(pars.free_names, ['d', 'f'])
        
    def tests_parameters_setvalues(self):
        pars = ParametersModel(['a', 'b', 'c'], [1, .2, .53],[True, False, True])
        pars['a'].value = 42.4
        self.assertEqual(pars['a'].value, 42.4)
        self.assertEqual(pars['b'].value, .2)
        self.assertEqual(pars['c'].value, .53)
        
        self.assertEqual(pars.free_values, [42.4, .53])
        
    
if __name__ == '__main__':
    unittest.main()
