""" Various tools for the Gaussian Process module."""
import jax.numpy as jnp

# constants
# TYPE_NUMBER = (float,int,jnp.number)
from typing import Union
TYPE_NUMBER = Union[float,int,jnp.number]


TABLE_LENGTH = 76
HEADER_PARAMETERS = "{Component:<4} {ID:<4} {Name:<15} {Value:<14} {Status:<9} {Linked:<9} {Type:<15} "

def sanity_checks(array_A, array_B):
    """ Check if the arrays are of the same shape 

    Parameters
    ----------
    array_A: (n,1) :obj:`jax.Array`
        First array.
    array_B: (n,1) :obj:`jax.Array`
        Second array.
    """
    assert jnp.shape(array_A) == jnp.shape(array_B), "The arrays must have the same shape."
        
def reshape_array(array):
    """ Reshape the array to a 2D array with jnp.shape(array,(len(array),1).

    Parameters
    ----------
    array: (n,) :obj:`jax.Array`
    
    Returns
    -------
    array: (n,1) :obj:`jax.Array`
        Reshaped array.

    """
    return jnp.reshape(array, (len(array), 1))
