""" Various tools for the Gaussian Process module."""


def check_instance(list_of_obj, classinfo):
    """Check if a list of objects is an instance of a class or a subclass of a class.
    
    Parameters
    ----------
    list_of_obj : list of objects
        List of objects to check.
    classinfo : class
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
