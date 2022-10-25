from .parameters import ParametersCovFunction
from scipy.spatial.distance import cdist


class CovarianceFunction:
    """ Master class for covariance functions.
    
    bridge between the parameters and the covariance function.
    
    """
    
    def __init__(self, params: ParametersCovFunction):
        #self.isotropic = isotropic
        self.parameters = params
    
    @classmethod
    def __classname(cls):
        return cls.__name__

    def print_info(self):
        print(f"Covariance function: {self.__classname()}")
        self.parameters.print_parameters()
        


def EuclideanDistance(xq, xp):
    """Compute the Euclidian distance between two arrays.
    
    using scipy.spatial.distance.cdist as it seems faster than a homemade version

    Parameters
    ----------
    xq : array of shape (n, 1)

    xp : array of shape (m, 1)

    Returns
    -------
    array of shape (n, m)
    """
    return cdist(xq, xp, metric='euclidean')
