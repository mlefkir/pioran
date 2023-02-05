"""Generic class for the covariance function.
"""

from dataclasses import dataclass
import jax.numpy as jnp
from .parameters import ParametersModel
from .utils import EuclideanDistance


@dataclass(slots=True)
class CovarianceFunction:
    """Master class for covariance functions.

    Bridge between the parameters and the covariance function. The covariance functions
    inherit from this class.
    
    Parameters
    ----------
    parameters_values : :obj:`ParametersModel` or list of float
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
        If parameters_values is not a list of float or a :obj:`ParametersModel`.

    Attributes
    ----------
    parameters : :obj:`ParametersModel`
        Parameters of the covariance function.

    Methods
    -------
    __init__
        Constructor of the covariance function class.
    __str__
        String representation of the covariance function.
        Include the representation of the parameters.
    get_cov_matrix
        Returns the covariance matrix.

    """    
    
    def __init__(self, parameters_values, names, boundaries, free_parameters):
        """Constructor of the covariance function inherited from the CovarianceFunction class.



        """
        #self.isotropic = isotropic
        # initialise the parameters
        if isinstance(parameters_values, ParametersModel):
            self.parameters = parameters_values
        elif isinstance(parameters_values, list) or isinstance(parameters_values, jnp.ndarray):
            self.parameters = ParametersModel(
                parameters_values, names=names, boundaries=boundaries, free_parameters=free_parameters)
        else:
            raise TypeError(
                "The parameters of the covariance function must be a list of floats or jnp.ndarray or a ParametersModel object.")

    @classmethod
    def __classname(cls):
        """Return the name of the class.
        
        Returns
        -------
        str
            Name of the class.
        """
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
    
    
    def get_cov_matrix(self, xq, xp):
        """Compute the covariance matrix between two arrays xq, xp.

        The term (xq-xp) is computed using the Euclidean distance.

        Parameters
        ----------
        xq : array of shape (n,1)
            First array.
        xp : array of shape (m,1)
            Second array.

        Returns
        -------
        K : array of shape (n,m)
            Covariance matrix.
        """
        # Compute the Euclidean distance between the query and the points
        dist = EuclideanDistance(xq, xp)
        # Compute the covariance matrix
        covMat = self.calculate(dist)
        
        return covMat
