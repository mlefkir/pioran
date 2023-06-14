

import unittest
import sys
sys.path.append('../../src')

from pioran.parameters import ParametersModel


class TestParameterModels(unittest.TestCase):
    def test_parameters_init(self):
        """
        Test the initialization of the ParametersModel class
        """
        pars = ParametersModel(['a', 'b', 'c'], [1, 2, 3],[True, True, True])
        self.assertEqual(pars.names, ['a', 'b', 'c'])
        self.assertEqual(pars.values, [1, 2, 3])
        self.assertEqual(pars.free_parameters, [True, True, True])
        self.assertEqual(pars.IDs, [1, 2, 3])
        self.assertEqual(pars.hyperparameters, [True, True, True])
        self.assertEqual(pars.components, [1, 1, 1])
        self.assertEqual(pars.relations, [None, None, None])

    def test_parameters_increment_IDs(self):
        """
        Test the increment_IDs method of the ParametersModel class
        """
        
        pars = ParametersModel(['a', 'b', 'c'], [1, 2, 3],[True, True, True])
        self.assertEqual(pars.IDs, [1, 2, 3])
        pars.increment_IDs(4)
        self.assertEqual(pars.IDs, [5, 6, 7])
        
    def test_parameters_increment_component(self):
        """
        Test the increment_component method of the ParametersModel class
        """
        
        pars = ParametersModel(['a', 'b', 'c'], [1, 2, 3],[True, True, True])
        self.assertEqual(pars.components, [1, 1, 1])
        pars.increment_component(4)
        self.assertEqual(pars.components, [5, 5, 5])
        
    def test_parameters_append(self):
        """
        Test the append method of the ParametersModel class
        """
        
        pars = ParametersModel(['a', 'b', 'c'], [1, 2, 3],[True, True, True])
        pars.append('d', 4, True)
        self.assertEqual(pars.names, ['a', 'b', 'c', 'd'])
        self.assertEqual(pars.values, [1, 2, 3, 4])
        self.assertEqual(pars.free_parameters, [True, True, True, True])
        self.assertEqual(pars.IDs, [1, 2, 3, 4])
        self.assertEqual(pars.hyperparameters, [True, True, True, True])
        self.assertEqual(pars.components, [1, 1, 1, 1])
        self.assertEqual(pars.relations, [None, None, None, None])   
        
        pars2 = ParametersModel(['a', 'b', 'c'], [1, 2, 3],[True, True, True])
        pars2.append('d', 4, True, hyperparameter=False, component=6, relation='testingrelation')
        pars2.append('e', 5, False, ID=5, hyperparameter=False, component=7, relation='test')
        self.assertEqual(pars2.names, ['a', 'b', 'c', 'd', 'e'])
        self.assertEqual(pars2.values, [1, 2, 3, 4, 5])
        self.assertEqual(pars2.free_parameters, [True, True, True, True, False])
        self.assertEqual(pars2.IDs, [1, 2, 3, 4, 5])
        self.assertEqual(pars2.hyperparameters, [True, True, True, False, False])
        self.assertEqual(pars2.components, [1, 1, 1, 6, 7])
        self.assertEqual(pars2.relations, [None, None, None, 'testingrelation', 'test'])
    
    def test_parameters_setnames(self):
        pars = ParametersModel(['a', 'b', 'c'], [1, 2, 3],[True, True, True])
        pars.set_names(['d', 'e', 'f'])
        self.assertEqual(pars.names, ['d', 'e', 'f'])

    def test_parameters_freenames(self):
        pars = ParametersModel(['a', 'b', 'c'], [1, 2, 3],[True, False, True])
        self.assertEqual(pars.free_names, ['a', 'c'])
        
                
    def test_parameters_set_free_values(self):
        pars = ParametersModel(['a', 'b', 'c'], [1, 2, 3],[True, True, True])
        pars.set_free_values([4, 5, 6])
        self.assertEqual(pars.values, [4, 5, 6])
        self.assertEqual(pars.free_values, [4, 5, 6])
        
        pars = ParametersModel(['a', 'b', 'c'], [1, 2, 3],[True, False, False])
        pars.set_free_values([4454])
        self.assertEqual(pars.values, [4454, 2, 3])
        self.assertEqual(pars.free_values, [4454])
        
        
        
    
if __name__ == '__main__':
    unittest.main()
