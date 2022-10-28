"""Generic class and functions for the covariance functions
"""
import numpy as np
from .parameters import ParametersCovFunction
from .utils import EuclideanDistance

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
    __str__()
        String representation of the covariance function.
        Include the representation of the parameters.


    """

    def __init__(self, parameters_values, names, boundaries, free_parameters):
        """Constructor of the squared exponential covariance function inherited from the CovarianceFunction class.

        Parameters
        ----------
        parameters_values : list of float or ParametersCovFunction
            Values of the parameters of the covariance function.
        names : list of str
            Names of the parameters of the covariance function.
        boundaries : list of (list of float or list of None)
            Boundaries of the parameters of the covariance function.    
        free_parameters : list of bool
            List of bool to indicate if the parameters are free or not.

        Raises
        ------
        TypeError
            If the parameters_values is not a list of float or a ParametersCovFunction.

        """
        #self.isotropic = isotropic
        # initialise the parameters
        if isinstance(parameters_values, ParametersCovFunction):
            self.parameters = parameters_values
        elif isinstance(parameters_values, list) or isinstance(parameters_values, np.ndarray):
            self.parameters = ParametersCovFunction(
                parameters_values, names=names, boundaries=boundaries, free_parameters=free_parameters)
        else:
            raise TypeError(
                "The parameters of the covariance function must be a list of floats or np.ndarray or a ParametersCovFunction object.")

    @classmethod
    def __classname(cls):
        """Return the name of the class."""
        return cls.__name__

    def __str__(self):
        """String representation of the covariance function.

        Returns
        -------
        str
            String representation of the covariance function.
            Include the representation of the parameters.
        """
        s = f"Covariance function: {self.__classname()}\n"
        s += f"Number of parameters: {len(self.parameters)}\n"
        s += self.parameters.__str__()
        return s
    
