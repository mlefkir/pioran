""" Various tools for the Gaussian Process module."""
import jax.numpy as jnp

# constants
# TYPE_NUMBER = (float,int,jnp.number)
from typing import Union
TYPE_NUMBER = Union[float,int,jnp.number]


TABLE_LENGTH = 80+30+4
HEADER_PARAMETERS = "{ID:<4} {Name:<15} {Value:<14} {Min:<10} {Max:<10} {Status:<9} {Linked:<9} {Expression:<15} {Type:<15} "


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
