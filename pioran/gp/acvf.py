"""Collection of classes for covariance functions

- Exponential
- Squared Exponential
- Matern 3/2
- Matern 5/2
- Rational Quadratic
"""

import jax.numpy as jnp
from .acvf_base import CovarianceFunction
from .utils import EuclideanDistance
 


class Exponential(CovarianceFunction):
    """ Class for the exponential covariance function.

    K(r) = variance * exp( - r / length)

    Attributes
    ----------
    parameters: ParametersModel
        Parameters of the covariance function.

    Methods
    -------
    get_cov_matrix:
        Compute the covariance matrix between two arrays.
    """

    def __init__(self, parameters_values, **kwargs):
        """Constructor of the covariance function inherited from the CovarianceFunction class.

        Parameters
        ----------
        parameters_values: list of float or ParametersModel
            Values of the parameters of the covariance function.
        **kwargs: dict
            Arguments for the ParametersModel class.
            free_parameters: list of bool
                List of bool to indicate if the parameters are free or not.
        """
        assert len(parameters_values) == 2, 'The number of parameters for this  covariance function must be 2'
        free_parameters = kwargs.get('free_parameters', [True, True])
        # initialise the parameters and check
        CovarianceFunction.__init__(self, parameters_values, names=['variance', 'length'], boundaries=[[0, jnp.inf], [0, jnp.inf]], free_parameters=free_parameters)

    def get_cov_matrix(self, xq, xp):
        """Compute the covariance matrix between two arrays for the exponential covariance function.

        K(xq,xp) = variance * exp( - (xq-xp) / length )

        The term (xq-xp) is computed using the Euclidean distance from the module covarfun.distance

        Parameters
        ----------
        xq: array of shape (n,1)
            First array.
        xp: array  of shape (m,1)
            Second array.

        Returns
        -------
        K: array of shape (n,m)
            Covariance matrix.
        """
        # Compute the Euclidean distance between the query and the points
        dist = EuclideanDistance(xq, xp)
        # Compute the covariance matrix
        covMat = self.calculate(dist)

        return covMat
    
    
    def calculate(self,x):
        """Compute the autocovariance for an array of lags

        Parameters
        ----------
        x : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        
        return  self.parameters['variance'].value * jnp.exp(- jnp.abs(x) * self.parameters['length'].value)

class SquareExponential(CovarianceFunction):
    """ Class for the squared exponential covariance function.

    K(r) = variance * exp( -1/2 * r^2 / length^2)

    Attributes
    ----------
    parameters: ParametersModel
        Parameters of the covariance function.

    Methods
    -------
    get_cov_matrix:
        Compute the covariance matrix between two arrays.
    """

    def __init__(self, parameters_values, **kwargs):
        """Constructor of the covariance function inherited from the CovarianceFunction class.

        Parameters
        ----------
        parameters_values: list of float or ParametersModel
            Values of the parameters of the covariance function.
        **kwargs: dict
            Arguments for the ParametersModel class.
            free_parameters: list of bool
                List of bool to indicate if the parameters are free or not.
        """
        assert len(parameters_values) == 2, 'The number of parameters for this covariance function must be 2'

        free_parameters = kwargs.get('free_parameters', [True, True])
        # initialise the parameters and check
        CovarianceFunction.__init__(self, parameters_values, names=['variance', 'length'], boundaries=[[0, jnp.inf], [0, jnp.inf]], free_parameters=free_parameters)

    def get_cov_matrix(self, xq, xp):
        """Compute the covariance matrix between two arrays for the square exponential covariance function.

        K(xq,xp) = variance * exp( -1/2 * (xq-xp)^2 / length^2)

        The term (xq-xp) is computed using the Euclidean distance from the module covarfun.distance

        Parameters
        ----------
        xq: array of shape (n,1)
            First array.
        xp: array  of shape (m,1)
            Second array.

        Returns
        -------
        K: array of shape (n,m)
            Covariance matrix.
        """
        # Compute the Euclidean distance between the query and the points
        dist = EuclideanDistance(xq, xp)
        # Compute the covariance matrix
        covMat = self.calculate(dist)
        
        return covMat
    
    def calculate(self,x):
            """Compute the autocovariance for an array of lags

            Parameters
            ----------
            x : _type_
                _description_

            Returns
            -------
            _type_
                _description_
            """
            
            # return  self.parameters['variance'].value * jnp.exp(-0.5 * x**2 / self.parameters['length'].value**2)
            return  self.parameters['variance'].value * jnp.exp(-2 * jnp.pi**2 * x**2 * self.parameters['length'].value**2)

class Matern32(CovarianceFunction):
    """ Class for the Matern 3/2 covariance function.

    K(r) = variance * (1 + sqrt(3) * r / length) * exp( -sqrt(3) * r / length)

    Attributes
    ----------
    parameters: ParametersModel
        Parameters of the covariance function.

    Methods
    -------
    get_cov_matrix:
        Compute the covariance matrix between two arrays.
    """

    def __init__(self, parameters_values, **kwargs):
        """Constructor of the covariance function inherited from the CovarianceFunction class.
        
        Parameters
        ----------
        parameters_values: list of float or ParametersModel
            Values of the parameters of the covariance function.
        **kwargs: dict
            Arguments for the ParametersModel class.
            free_parameters: list of bool
                List of bool to indicate if the parameters are free or not.
        """
        assert len(parameters_values) == 2, 'The number of parameters for this covariance function must be 2'
        free_parameters = kwargs.get('free_parameters', [True, True])
        # initialise the parameters and check
        CovarianceFunction.__init__(self, parameters_values, names=['variance', 'length'], boundaries=[[0, jnp.inf], [0, jnp.inf]], free_parameters=free_parameters)

    def get_cov_matrix(self, xq, xp):
        """Compute the covariance matrix between two arrays for the Matern 3/2 covariance function.

        K(xq,xp) = variance *  (1 + sqrt(3) * (xq-xp) / length) * exp( -sqrt(3) * (xq-xp)/ length)

        The term (xq-xp) is computed using the Euclidean distance.

        Parameters
        ----------
        xq: array of shape (n,1)
            First array.
        xp: array  of shape (m,1)
            Second array.

        Returns
        -------
        K: array of shape (n,m)
            Covariance matrix.
        """
        # Compute the Euclidean distance between the query and the points
        dist = EuclideanDistance(xq, xp)
        # Compute the covariance matrix
        covMat = self.parameters['variance'].value * (1 + jnp.sqrt(3) * dist / self.parameters['length'].value ) * jnp.exp(-jnp.sqrt(3) * dist / self.parameters['length'].value )
        return covMat


class Matern52(CovarianceFunction):
    """ Class for the Matern 5/2 covariance function.

    K(r) = variance * (1 + sqrt(5) * r / length + 5 * r^2 / (3 * length^2) ) * exp( -sqrt(5) * r / length)

    Attributes
    ----------
    parameters: ParametersModel
        Parameters of the covariance function.

    Methods
    -------
    get_cov_matrix:
        Compute the covariance matrix between two arrays.
    """

    def __init__(self, parameters_values, **kwargs):
        """Constructor of the covariance function inherited from the CovarianceFunction class.

        Parameters
        ----------
        parameters_values: list of float or ParametersModel
            Values of the parameters of the covariance function.
        **kwargs: dict
            Arguments for the ParametersModel class.
            free_parameters: list of bool
                List of bool to indicate if the parameters are free or not.
        """
        assert len(parameters_values) == 2, 'The number of parameters for this covariance function must be 2'
        free_parameters = kwargs.get('free_parameters', [True, True])
        # initialise the parameters and check
        CovarianceFunction.__init__(self, parameters_values, names=['variance', 'length'], boundaries=[[0, jnp.inf], [0, jnp.inf]], free_parameters=free_parameters)

    def get_cov_matrix(self, xq, xp):
        """Compute the covariance matrix between two arrays for the Matern 5/2 covariance function.

        K(xq,xp) = variance *  (1 + sqrt(5) * (xq-xp) / length  + 5 * (xq-xp)^2 / (3 * length^2) ) * exp( -sqrt(5) * (xq-xp)/ length)

        The term (xq-xp) is computed using the Euclidean distance.

        Parameters
        ----------
        xq: array of shape (n,1)
            First array.
        xp: array  of shape (m,1)
            Second array.

        Returns
        -------
        K: array of shape (n,m)
            Covariance matrix.
        """
        # Compute the Euclidean distance between the query and the points
        dist = EuclideanDistance(xq, xp)
        # Compute the covariance matrix
        covMat = self.parameters['variance'].value * (1 + jnp.sqrt(5) * dist / self.parameters['length'].value + 5 * dist**2 / ( 3 * self.parameters['length'].value**2) ) * jnp.exp( - jnp.sqrt(5) * dist / self.parameters['length'].value )
        return covMat


class RationalQuadratic(CovarianceFunction):
    """ Class for the rational quadratic covariance function.

    K(r) = variance * (1 + r^2 / (2 * alpha * length^2) )^(-alpha)

    with: alpha, length > 0

    Attributes
    ----------
    parameters: ParametersModel
        Parameters of the covariance function.

    Methods
    -------
    get_cov_matrix:
        Compute the covariance matrix between two arrays.
    """
    
    def __init__(self, parameters_values, **kwargs):
        """Constructor of the covariance function inherited from the CovarianceFunction class.

        Parameters
        ----------
        parameters_values: list of float or ParametersModel
            Values of the parameters of the covariance function.
        **kwargs: dict
            Arguments for the ParametersModel class.
            free_parameters: list of bool
                List of bool to indicate if the parameters are free or not.
        """
        free_parameters = kwargs.get('free_parameters', [True, True, True])
        # initialise the parameters
        assert len(parameters_values) == 3, 'The number of parameters for the rational quadratic covariance function is 3.'
        CovarianceFunction.__init__(self, parameters_values, names=[ 'variance', 'alpha', 'length'], boundaries=[[0, jnp.inf], [0, jnp.inf], [0, jnp.inf]], free_parameters=free_parameters)

    def get_cov_matrix(self, xq, xp):
        """Compute the covariance matrix between two arrays for the rational quadratic covariance function.

        K(xq,xp) = variance *  (1 + (xq-xp)^2 / (2 * alpha * length^2) ) ^(-alpha)

        The term (xq-xp) is computed using the Euclidean distance.

        Parameters
        ----------
        xq: array of shape (n,1)
            First array.
        xp: array  of shape (m,1)
            Second array.

        Returns
        -------
        K: array of shape (n,m)
            Covariance matrix.
        """
        # Compute the Euclidean distance between the query and the points
        dist = EuclideanDistance(xq, xp)
        # Compute the covariance matrix
        covMat = self.parameters['variance'].value * (1 + dist**2 / ( 2 * self.parameters['alpha'].value * self.parameters['length'].value**2) ) ** ( - self.parameters['alpha'].value)
        return covMat