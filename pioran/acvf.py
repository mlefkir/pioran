"""Collection of classes for covariance functions

- Exponential
- Squared Exponential
- Matern 3/2
- Matern 5/2
- Rational Quadratic
"""
from jax import jit
import jax.numpy as jnp
from .acvf_base import CovarianceFunction
from .utils import EuclideanDistance
from .parameters import ParametersModel

class Exponential(CovarianceFunction):
    r"""Class for the exponential covariance function.

    .. math:: :label: expocov  
    
       K(\tau) = A \times \exp( {- |\tau| / \gamma}) 
       
    with     .. math::`A\gtr0` and   .. math::`\gamm>0`
    
    
    Parameters
    ----------
    parameters_values: list of float or ParametersModel
        Values of the parameters of the covariance function.
    **kwargs: dict
        Arguments for the ParametersModel class.
        
        free_parameters: list of bool
            List of bool to indicate if the parameters are free or not.
            
    Attributes
    ----------
    parameters: ParametersModel
        Parameters of the covariance function.

    """

    def __init__(self, parameters_values, **kwargs):
        """Constructor of the covariance function inherited from the CovarianceFunction class.
        """
        assert len(parameters_values) == 2, 'The number of parameters for this  covariance function must be 2'
        free_parameters = kwargs.get('free_parameters', [True, True])
        # initialise the parameters and check
        CovarianceFunction.__init__(self, parameters_values, names=['variance', 'length'], boundaries=[[0, jnp.inf], [0, jnp.inf]], free_parameters=free_parameters)
    
    def calculate(self,t):
        """Computes the exponential covariance function for an array of lags t.
        
        The expression is given by :math:numref:`expocov`
        with A and gamma positive reals.

        Parameters
        ----------
        t : JAXArray
            Array of lags.

        Returns
        -------
        covariance function evaluated on the array of lags.
        """
        
        return  self.parameters['variance'].value * jnp.exp(- jnp.abs(t) * self.parameters['length'].value)

class SquareExponential(CovarianceFunction):
    r""" Class for the squared exponential covariance function.

    .. math:: :label: squareexpo  

        K(\tau) = A \times \exp{\left( -\dfrac{\tau^2}{2\sigma}\right) } 


    Parameters
    ----------
    parameters_values: list of float or ParametersModel
        Values of the parameters of the covariance function.
    **kwargs: dict
        Arguments for the ParametersModel class.
        free_parameters: list of bool
            List of bool to indicate if the parameters are free or not.

    Attributes
    ----------
    parameters: ParametersModel
        Parameters of the covariance function.

    """

    def __init__(self, parameters_values, **kwargs):
        """Constructor of the covariance function inherited from the CovarianceFunction class. """
        assert len(parameters_values) == 2, 'The number of parameters for this covariance function must be 2'

        free_parameters = kwargs.get('free_parameters', [True, True])
        # initialise the parameters and check
        CovarianceFunction.__init__(self, parameters_values, names=['variance', 'length'], boundaries=[[0, jnp.inf], [0, jnp.inf]], free_parameters=free_parameters)

    def calculate(self,x):
        """Compute the covariance for an array of lags.

        The expression is given by :math:numref:`squareexpo`
        with A and gamma positive reals.
        
        Parameters
        ----------
        x : Array
            Array to calculate the covariance.

        Returns
        -------
        Covariance array.
        """
        
        # return  self.parameters['variance'].value * jnp.exp(-0.5 * x**2 / self.parameters['length'].value**2)
        return  self.parameters['variance'].value * jnp.exp(-2 * jnp.pi**2 * x**2 * self.parameters['length'].value**2)

class Matern32(CovarianceFunction):
    r""" Class for the Matern 3/2 covariance function.

    .. math:: :label: expocov  
    
       K(\tau) = A \times (1+\dfrac{ \sqrt{3} \tau}{\sigma} \exp( {-  \sqrt{3} |\tau| / \gamma}) 
       
    K(r) = variance * (1 + sqrt(3) * r / length) * exp( -sqrt(3) * r / length)

    Parameters
    ----------
    parameters_values: list of float or ParametersModel
        Values of the parameters of the covariance function.
    **kwargs: dict
        Arguments for the ParametersModel class.
        free_parameters: list of bool
            List of bool to indicate if the parameters are free or not.
                
    Attributes
    ----------
    parameters: ParametersModel
        Parameters of the covariance function.
        
    """

    def __init__(self, parameters_values, **kwargs):
        """Constructor of the covariance function inherited from the CovarianceFunction class.

        """
        assert len(parameters_values) == 2, 'The number of parameters for this covariance function must be 2'
        free_parameters = kwargs.get('free_parameters', [True, True])
        # initialise the parameters and check
        CovarianceFunction.__init__(self, parameters_values, names=['variance', 'length'], boundaries=[[0, jnp.inf], [0, jnp.inf]], free_parameters=free_parameters)

    def calculate(self,t):
        return self.parameters['variance'].value * (1 + jnp.sqrt(3) * t / self.parameters['length'].value ) * jnp.exp(-jnp.sqrt(3) * t / self.parameters['length'].value )


class Matern52(CovarianceFunction):
    r""" Class for the Matern 5/2 covariance function.

    K(r) = variance * (1 + sqrt(5) * r / length + 5 * r^2 / (3 * length^2) ) * exp( -sqrt(5) * r / length)

    Parameters
    ----------
    parameters_values: list of float or ParametersModel
        Values of the parameters of the covariance function.
    **kwargs: dict
        Arguments for the ParametersModel class.
        free_parameters: list of bool
            List of bool to indicate if the parameters are free or not.

    Attributes
    ----------
    parameters: ParametersModel
        Parameters of the covariance function.

    """

    def __init__(self, parameters_values, **kwargs):
        """Constructor of the covariance function inherited from the CovarianceFunction class.
        """
        assert len(parameters_values) == 2, 'The number of parameters for this covariance function must be 2'
        free_parameters = kwargs.get('free_parameters', [True, True])
        # initialise the parameters and check
        CovarianceFunction.__init__(self, parameters_values, names=['variance', 'length'], boundaries=[[0, jnp.inf], [0, jnp.inf]], free_parameters=free_parameters)

    def calculate(self,x):

        return self.parameters['variance'].value * (1 + jnp.sqrt(5) * x / self.parameters['length'].value + 5 * x**2 / ( 3 * self.parameters['length'].value**2) ) * jnp.exp( - jnp.sqrt(5) * x / self.parameters['length'].value )

class RationalQuadratic(CovarianceFunction):
    r""" Class for the rational quadratic covariance function.

    K(r) = variance * (1 + r^2 / (2 * alpha * length^2) )^(-alpha)

    with: alpha, length > 0

    Parameters
    ----------
    parameters_values: list of float or ParametersModel
        Values of the parameters of the covariance function.
    **kwargs: dict
        Arguments for the ParametersModel class.
        free_parameters: list of bool
            List of bool to indicate if the parameters are free or not.

    Attributes
    ----------
    parameters: ParametersModel
        Parameters of the covariance function.

    """
    
    def __init__(self, parameters_values, **kwargs):
        """Constructor of the covariance function inherited from the CovarianceFunction class.
        """
        free_parameters = kwargs.get('free_parameters', [True, True, True])
        # initialise the parameters
        assert len(parameters_values) == 3, 'The number of parameters for the rational quadratic covariance function is 3.'
        CovarianceFunction.__init__(self, parameters_values, names=[ 'variance', 'alpha', 'length'], boundaries=[[0, jnp.inf], [0, jnp.inf], [0, jnp.inf]], free_parameters=free_parameters)

    def calculate(self,x):
        return self.parameters['variance'].value * (1 + x**2 / ( 2 * self.parameters['alpha'].value * self.parameters['length'].value**2) ) ** ( - self.parameters['alpha'].value)
