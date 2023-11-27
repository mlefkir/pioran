

import unittest
import sys
sys.path.append('../../src')

from pioran.parameters import Parameter


class TestParameter(unittest.TestCase):
    def test_parameter_init(self):
        """
        Test the initialization of the Parameter class.
        """
        
        p = Parameter('test', 1.0, free=True, ID=1, hyperparameter=True, component=1, relation=None)
        self.assertEqual(p.name, 'test')
        self.assertEqual(p.value, 1.0)
        self.assertEqual(p.free, True)
        self.assertEqual(p.ID, 1)
        self.assertEqual(p.hyperparameter, True)
        self.assertEqual(p.component, 1)
        self.assertEqual(p.relation, None)

        
    def test_parameter_set_value(self):
        """
        Test the set_value method of the Parameter class.
        """
        
        p = Parameter('test', 1.0, free=True, ID=1, hyperparameter=True, component=1, relation=None)
        p.value = 2.0 
        self.assertEqual(p.value, 2.0)
    
        p = Parameter('test', 1, free=True, ID=1, hyperparameter=True, component=1, relation=None)
        p.value = 12.4564 
        self.assertEqual(p.value, 12.4564)
        
    def test_parameter_set_name(self):
        """
        Test the set_name method of the Parameter class.
        """
        
        p = Parameter('test', 1.0, free=True, ID=1, hyperparameter=True, component=1, relation=None)
        p.name = 'test2' 
        self.assertEqual(p.name, 'test2')
    
        
    def test_flatten_tree(self):
        p = Parameter('test', 1.0, free=True, ID=1, hyperparameter=True, component=1, relation=None)
        c,a = p.tree_flatten()
        if c == (): raise Exception('Children not set correctly')
        q = p.tree_unflatten(a,c)
        self.assertEqual(q.name, 'test')
        self.assertEqual(q.value, 1.0)        
    
if __name__ == '__main__':
    unittest.main()
