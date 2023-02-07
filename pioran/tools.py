""" Various tools for the Gaussian Process module."""
import jax.numpy as jnp

# constants
# TYPE_NUMBER = (float,int,jnp.number)
from typing import Union
TYPE_NUMBER = Union[float,int,jnp.number]


TABLE_LENGTH = 80+30+4
HEADER_PARAMETERS = "{ID:<4} {Name:<15} {Value:<14} {Min:<10} {Max:<10} {Status:<9} {Linked:<9} {Expression:<15} {Type:<15} "

def sanity_checks(array_A, array_B):
    """ Check if the lists are of the same shape 

    Parameters
    ----------
    array_A: array of shape (n,1)
        First array.
    array_B: array  of shape (m,1)
        Second array.
    """
    assert jnp.shape(array_A) == jnp.shape(array_B), "The arrays must have the same shape."
        
def reshape_array(array):
    """ Reshape the array to a 2D array with jnp.shape(array,(len(array),1).

    Parameters
    ----------
    array: 1D array
    
    Returns
    -------
    array: 2D array
        Reshaped array.

    """
    return jnp.reshape(array, (len(array), 1))


def check_instance(list_of_obj, classinfo):
    """Check if a list of objects is an instance of a class or a subclass of a class.
    
    Parameters
    ----------
    list_of_obj: list of objects
        List of objects to check.
    classinfo: class
        Class to check.
    
    Returns
    -------
    bool
        True if all the objects are from the same class, False otherwise.
    """
    
    for obj in list_of_obj:
        if not isinstance(obj, classinfo):
            return False
    return True
