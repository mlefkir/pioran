"""Generic class and functions for the covariance functions
"""
import numpy as np
from scipy.spatial.distance import cdist
from .parameters import ParametersCovFunction


class CovarianceFunction:
    """ Master class for covariance functions.

    Bridge between the parameters and the covariance function.

    Attributes
    ----------
    parameters : ParametersCovFunction
        Parameters of the covariance function.


    Methods
    -------
    __init__(parameters_values, names, boundaries)
        Constructor of the covariance function class.
    print_info()
        Print the information about the covariance function.



    """

    def __init__(self, parameters_values, names, boundaries):
        """Constructor of the squared exponential covariance function inherited from the CovarianceFunction class.

        Parameters
        ----------
        parameters_values : list of float or ParametersCovFunction
            Values of the parameters of the covariance function.

        Raises
        ------
        TypeError
            If the parameters_values is not a list of float or a ParametersCovFunction.

        """
        #self.isotropic = isotropic
        # initialise the parameters
        if isinstance(parameters_values, ParametersCovFunction):
            self.parameters = parameters_values
            self.parameters.update_names(names)
            self.parameters.update_boundaries(boundaries)
        elif isinstance(parameters_values, list) or isinstance(parameters_values, np.ndarray):
            self.parameters = ParametersCovFunction(
                parameters_values, names=names, boundaries=boundaries)
        else:
            raise TypeError(
                "The parameters of the covariance function must be a list of floats or np.ndarray or a ParametersCovFunction object.")

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
