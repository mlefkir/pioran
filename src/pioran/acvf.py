"""Collection of classes for covariance functions."""

import jax.numpy as jnp

from .acvf_base import CovarianceFunction
from .parameters import ParametersModel


class Exponential(CovarianceFunction):
    r"""Class for the exponential covariance function as defined below:

    .. math:: :label: expocov  
    
       K(\tau) = \dfrac{A}{2\gamma} \times \exp( {- |\tau| \gamma}).

    with the variance :math:`A\ge 0` and length :math:`\gamma>0`.
    
    The parameters are stored in the `parameters` attribute which is a :class:`ParametersModel` object. 
    The values of the parameters can be accessed using the `parameters` attribute via two keys: '`variance`' and '`length`'.
    
    The covariance function is evaluated on an array of lags :math:`\tau` using the `calculate` method.
    
    
    Parameters
    ----------
    param_values : :obj:`list of float`
        Values of the parameters of the covariance function.
    **kwargs : :obj:`dict`        
        free_parameters: :obj:`list of bool`
            List of bool to indicate if the parameters are free or not.
            
    Attributes
    ----------
    parameters : :obj:`ParametersModel`
        Parameters of the covariance function.
        
    Methods
    -------
    calculate(t)
        Computes the exponential covariance function for an array of lags :math:`\tau`.
    """
    
    parameters: ParametersModel
    expression = 'exponential'

    def __init__(self, param_values, **kwargs):
        """Constructor of the covariance function inherited from the CovarianceFunction class.
        """
        assert len(param_values) == 2, 'The number of parameters for this covariance function must be 2'
        free_parameters = kwargs.get('free_parameters', [True, True])
        CovarianceFunction.__init__(self, param_values=param_values,param_names=['variance', 'length'], free_parameters=free_parameters)
    
    def calculate(self,t) -> jnp.array:
        r"""Computes the exponential covariance function for an array of lags :math:`\tau`.
        
        The expression is given by Equation :math:numref:`expocov`.
        with the variance :math:`A\ge 0` and length :math:`\gamma>0`.

        Parameters
        ----------
        t : :obj:`jnp.array`
            Array of lags.

        Returns
        -------
        Covariance function evaluated on the array of lags.
        """
        
        # return  self.parameters['variance'].value * jnp.exp(- jnp.abs(t) / self.parameters['length'].value)
        return  0.5 * self.parameters['variance'].value / self.parameters['length'].value *  jnp.exp(- jnp.abs(t) * self.parameters['length'].value)

class SquaredExponential(CovarianceFunction):
    r""" Class for the squared exponential covariance function.

    .. math:: :label: exposquare  

        K(\tau) = A \times \exp{\left( -2 \pi^2 \tau^2 \sigma^2 \right)}. 

    with the variance :math:`A\ge 0` and length :math:`\sigma>0`.
    
    The parameters are stored in the `parameters` attribute which is a :class:`ParametersModel` object. 
    The values of the parameters can be accessed using the `parameters` attribute via two keys: '`variance`' and '`length`'.
    
    The covariance function is evaluated on an array of lags :math:`\tau` using the `calculate` method.
    
    
    Parameters
    ----------
    param_values : :obj:`list of float`
        Values of the parameters of the covariance function.
    **kwargs : :obj:`dict`        
        free_parameters: :obj:`list of bool`
            List of bool to indicate if the parameters are free or not.
            
    Attributes
    ----------
    parameters : :class:`ParametersModel`
        Parameters of the covariance function.
    
    Methods
    -------
    calculate(x)
        Computes the squared exponential covariance function for an array of lags :math:`\tau`.
    """
    parameters: ParametersModel
    expression = 'exponential_squared'

    def __init__(self, param_values, **kwargs):
        """Constructor of the covariance function inherited from the CovarianceFunction class. """
        assert len(param_values) == 2, 'The number of parameters for this covariance function must be 2'

        free_parameters = kwargs.get('free_parameters', [True, True])
        # initialise the parameters and check
        CovarianceFunction.__init__(self, param_values, param_names=['variance', 'length'], free_parameters=free_parameters)

    def calculate(self,t) -> jnp.array:
        r"""Compute the squared exponential covariance function for an array of lags :math:`\tau`.
      
        The expression is given by Equation :math:numref:`exposquare`.
        with the variance :math:`A\ge 0` and length :math:`\sigma>0`.

        Parameters
        ----------
        t : :obj:`jnp.array`
            Array of lags.

        Returns
        -------
        Covariance function evaluated on the array of lags.
        """
        
        return  self.parameters['variance'].value * jnp.exp(-2 * jnp.pi**2 * t**2 * self.parameters['length'].value**2)

class Matern32(CovarianceFunction):
    r""" Class for the Matern 3/2 covariance function.

    .. math:: :label: matern32  
    
       K(\tau) = A \times \left(1+\dfrac{ \sqrt{3} \tau}{\gamma} \right)  \exp{\left(-  \sqrt{3} |\tau| / \gamma \right)}. 
       
    with the variance :math:`A\ge 0` and length :math:`\gamma>0`

    The parameters are stored in the `parameters` attribute which is a :class:`ParametersModel` object. 
    The values of the parameters can be accessed using the `parameters` attribute via two keys: '`variance`' and '`length`'.
    
    The covariance function is evaluated on an array of lags :math:`\tau` using the `calculate` method.
    
    
    Parameters
    ----------
    param_values : :obj:`list of float`
        Values of the parameters of the covariance function.
    **kwargs : :obj:`dict`        
        free_parameters: :obj:`list of bool`
            List of bool to indicate if the parameters are free or not.
            
    Attributes
    ----------
    parameters : :class:`ParametersModel`
        Parameters of the covariance function.
    
    Methods
    -------
    calculate(t)
        Computes the Matern 3/2 covariance function for an array of lags :math:`\tau`.
    """
    parameters: ParametersModel
    expression = 'matern32'
    
    def __init__(self, param_values, **kwargs):
        """Constructor of the covariance function inherited from the CovarianceFunction class.
        """
        assert len(param_values) == 2, 'The number of parameters for this covariance function must be 2'
        free_parameters = kwargs.get('free_parameters', [True, True])
        # initialise the parameters and check
        CovarianceFunction.__init__(self, param_values, param_names=['variance', 'length'], free_parameters=free_parameters)

    def calculate(self,t) -> jnp.array:
        r"""Computes the Matérn 3/2 covariance function for an array of lags :math:`\tau`.
        
        The expression is given by Equation :math:numref:`matern32`.
        with the variance :math:`A\ge 0` and scale :math:`\gamma>0`.

        Parameters
        ----------
        t : `jnp.array`
            Array of lags.

        Returns
        -------
        Covariance function evaluated on the array of lags.
        """
        return self.parameters['variance'].value * (1 + jnp.sqrt(3) * t / self.parameters['length'].value ) * jnp.exp(-jnp.sqrt(3) * t / self.parameters['length'].value )


class Matern52(CovarianceFunction):
    r""" Class for the Matern 5/2 covariance function.

    .. math:: :label: matern52  
    
       K(\tau) = A \times \left(1+\dfrac{ \sqrt{5} \tau}{\gamma} + 5 \dfrac{\tau^2}{3\gamma^2} \right)  \exp{\left(-  \sqrt{5} |\tau| / \gamma \right) }. 
       
       
    with the variance :math:`A\ge 0` and length :math:`\gamma>0`.

    The parameters are stored in the `parameters` attribute which is a :class:`ParametersModel` object. 
    The values of the parameters can be accessed using the `parameters` attribute via two keys: '`variance`' and '`length`'.
    
    The covariance function is evaluated on an array of lags :math:`\tau` using the `calculate` method.
    
    
    Parameters
    ----------
    param_values : :obj:`list of float`
        Values of the parameters of the covariance function.
    **kwargs : :obj:`dict`        
        free_parameters: :obj:`list of bool`
            List of bool to indicate if the parameters are free or not.
            
    Attributes
    ----------
    parameters : :class:`ParametersModel`
        Parameters of the covariance function.
        
    Methods
    -------
    calculate(t)
        Computes the Matern 5/2 covariance function for an array of lags :math:`\tau`.
    """
    parameters: ParametersModel
    expression = 'matern52'
    
    

    def __init__(self, param_values, **kwargs):
        """Constructor of the covariance function inherited from the CovarianceFunction class.
        """
        assert len(param_values) == 2, 'The number of parameters for this covariance function must be 2'
        free_parameters = kwargs.get('free_parameters', [True, True])
        # initialise the parameters and check
        CovarianceFunction.__init__(self, param_values, param_names=['variance', 'length'], free_parameters=free_parameters)

    def calculate(self,t) -> jnp.array:
        r"""Computes the Matérn 5/2 covariance function for an array of lags :math:`\tau`.
        
        The expression is given by Equation :math:numref:`matern52`.
        with the variance :math:`A\ge 0` and scale :math:`\gamma>0`.

        Parameters
        ----------
        t : `jnp.array`
            Array of lags.

        Returns
        -------
        Covariance function evaluated on the array of lags.
        """
        return self.parameters['variance'].value * (1 + jnp.sqrt(5) * t / self.parameters['length'].value + 5 * t**2 / ( 3 * self.parameters['length'].value**2) ) * jnp.exp( - jnp.sqrt(5) * t / self.parameters['length'].value )

class RationalQuadratic(CovarianceFunction):
    r""" Class for the rational quadratic covariance function.


    .. math:: :label: rationalquadratic  
    
       K(\tau) = A \times {\left(1+ \dfrac{\tau^2}{2\alpha \gamma^2}  \right) }^{-\alpha}.
       
       
    with the variance :math:`A\ge 0`, length :math:`\gamma>0` and scale :math:`\alpha>0`

    The parameters are stored in the `parameters` attribute which is a :class:`ParametersModel` object. 
    The values of the parameters can be accessed using the `parameters` attribute via two keys: '`variance`' and '`length`'.
    
    The covariance function is evaluated on an array of lags :math:`\tau` using the `calculate` method.
    
    
    Parameters
    ----------
    param_values : :obj:`list of float`
        Values of the parameters of the covariance function.
    **kwargs : :obj:`dict`        
        free_parameters: :obj:`list of bool`
            List of bool to indicate if the parameters are free or not.
            
    Attributes
    ----------
    parameters : :class:`ParametersModel`
        Parameters of the covariance function.
        
    Methods
    -------
    calculate(t)
        Computes the rational quadratic covariance function for an array of lags :math:`\tau`.
    """
    parameters: ParametersModel
    expression = 'rationalquadratic'
    
    
    def __init__(self, param_values, **kwargs):
        """Constructor of the covariance function inherited from the CovarianceFunction class.
        """
        free_parameters = kwargs.get('free_parameters', [True, True, True])
        # initialise the parameters
        assert len(param_values) == 3, 'The number of parameters for the rational quadratic covariance function is 3.'
        CovarianceFunction.__init__(self, param_values, param_names=[ 'variance', 'alpha', 'length'], free_parameters=free_parameters)

    def calculate(self,x) -> jnp.array:
        r"""Computes the rational quadratic covariance function for an array of lags :math:`\tau`.
        
        The expression is given by Equation :math:numref:`rationalquadratic`.
        with the variance :math:`A\ge 0`, length :math:`\gamma>0` and scale :math:`\alpha>0`.

        Parameters
        ----------
        t : `jnp.array`
            Array of lags.

        Returns
        -------
        Covariance function evaluated on the array of lags.
        """        
        return self.parameters['variance'].value * (1 + x**2 / ( 2 * self.parameters['alpha'].value * self.parameters['length'].value**2) ) ** ( - self.parameters['alpha'].value)
