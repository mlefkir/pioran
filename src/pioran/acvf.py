"""Collection of classes for covariance functions."""
import jax
import jax.numpy as jnp

from .acvf_base import CovarianceFunction
from .parameters import ParametersModel
from .tools import Array_type
from .utils.carma_utils import (lorentzians_to_roots, quad_to_roots,
                                roots_to_quad)


class Exponential(CovarianceFunction):
    r"""Class for the exponential covariance function.
    
    .. math:: :label: expocov  
    
       K(\tau) = \dfrac{A}{2\gamma} \times \exp( {- |\tau| \gamma}).

    with the variance :math:`A\ge 0` and length :math:`\gamma>0`.
    
    The parameters are stored in the `parameters` attribute which is a :class:`~pioran.parameters.ParametersModel` object. 
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
    parameters : :class:`~pioran.parameters.ParametersModel`
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
    
    def calculate(self,t) -> jax.Array:
        r"""Computes the exponential covariance function for an array of lags :math:`\tau`.
        
        The expression is given by Equation :math:numref:`expocov`.
        with the variance :math:`A\ge 0` and length :math:`\gamma>0`.

        Parameters
        ----------
        t : :obj:`jax.Array`
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
    
    The parameters are stored in the `parameters` attribute which is a :class:`~pioran.parameters.ParametersModel` object. 
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
    parameters : :class:`~pioran.parameters.ParametersModel`
        Parameters of the covariance function.
    
    Methods
    -------
    calculate(x)
        Computes the squared exponential covariance function for an array of lags :math:`\tau`.
    """
    parameters: ParametersModel
    expression = 'squared_exponential'

    def __init__(self, param_values, **kwargs):
        """Constructor of the covariance function inherited from the CovarianceFunction class. """
        assert len(param_values) == 2, 'The number of parameters for this covariance function must be 2'

        free_parameters = kwargs.get('free_parameters', [True, True])
        # initialise the parameters and check
        CovarianceFunction.__init__(self, param_values, param_names=['variance', 'length'], free_parameters=free_parameters)

    def calculate(self,t) -> jax.Array:
        r"""Compute the squared exponential covariance function for an array of lags :math:`\tau`.
      
        The expression is given by Equation :math:numref:`exposquare`.
        with the variance :math:`A\ge 0` and length :math:`\sigma>0`.

        Parameters
        ----------
        t : :obj:`jax.Array`
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

    The parameters are stored in the `parameters` attribute which is a :class:`~pioran.parameters.ParametersModel` object. 
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
    parameters : :class:`~pioran.parameters.ParametersModel`
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

    def calculate(self,t) -> jax.Array:
        r"""Computes the Matérn 3/2 covariance function for an array of lags :math:`\tau`.
        
        The expression is given by Equation :math:numref:`matern32`.
        with the variance :math:`A\ge 0` and scale :math:`\gamma>0`.

        Parameters
        ----------
        t : :obj:`jax.Array`
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

    The parameters are stored in the `parameters` attribute which is a :class:`~pioran.parameters.ParametersModel` object. 
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
    parameters : :class:`~pioran.parameters.ParametersModel`
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

    def calculate(self,t) -> jax.Array:
        r"""Computes the Matérn 5/2 covariance function for an array of lags :math:`\tau`.
        
        The expression is given by Equation :math:numref:`matern52`.
        with the variance :math:`A\ge 0` and scale :math:`\gamma>0`.

        Parameters
        ----------
        t : :obj:`jax.Array`
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

    The parameters are stored in the `parameters` attribute which is a :class:`~pioran.parameters.ParametersModel` object. 
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
    parameters : :class:`~pioran.parameters.ParametersModel`
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

    def calculate(self,x) -> jax.Array:
        r"""Computes the rational quadratic covariance function for an array of lags :math:`\tau`.
        
        The expression is given by Equation :math:numref:`rationalquadratic`.
        with the variance :math:`A\ge 0`, length :math:`\gamma>0` and scale :math:`\alpha>0`.

        Parameters
        ----------
        t : :obj:`jax.Array`
            Array of lags.
            
        Returns
        -------
        Covariance function evaluated on the array of lags.
        """        
        return self.parameters['variance'].value * (1 + x**2 / ( 2 * self.parameters['alpha'].value * self.parameters['length'].value**2) ) ** ( - self.parameters['alpha'].value)

class CARMA_covariance(CovarianceFunction):
    r"""Base class for the covariance function of a Continuous AutoRegressive Moving Average (CARMA) process.
    

    Parameters
    ----------
    p : :obj:`int`
        Order of the AR part of the model.
    q : :obj:`int`
        Order of the MA part of the model. 0 <= q < p
        
    Attributes
    ----------
    parameters : :obj:`ParametersModel`
        Parameters of the model.
    p : :obj:`int`
        Order of the AR part of the model.
    q : :obj:`int`
        Order of the MA part of the model. 0 <= q < p
    _p : :obj:`int`
        Order of the AR part of the model. p+1
    _q : :obj:`int`
        Order of the MA part of the model. q+1   
    
    
    """
    
    parameters: ParametersModel
    expression: str
    p: int
    q: int
    _p: int
    _q: int
    
    def __init__(self, p, q,AR_quad=None, beta=None, lorentzian_centroids=None, lorentzian_widths=None,weights=None,**kwargs) -> None:
        """Constructor method
        """
        sigma = kwargs.get("sigma",1)
        
        CovarianceFunction.__init__(self, param_values=[sigma],param_names=['sigma'], free_parameters=[True])
        
        self.p = p
        self.q = q
        assert self.q < self.p, "q must be smaller than p"
        self.expression = f'CARMA({p},{q})'
        self._p = p+1
        self._q = q+1
        
        
        # if we provide the quadratic coefficients 
        if AR_quad is not None:
            
            # set the AR parameters
            if isinstance(AR_quad,Array_type):
                assert len(AR_quad) == self.p, "AR_quad must have length p"
                for i,ar in enumerate(AR_quad):
                    self.parameters.append(f"a_{i+1}",ar,True,hyperparameter=True)
            else:
                assert self.p == 1, "p must be 1 if AR_quad is not an array"
                self.parameters.append(f"a_1",AR_quad,True,hyperparameter=True)
            
            # set the MA parameters
            
            if self.q == 0:
                assert beta is None, "beta must be None if q = 0"
            self.parameters.append(f"beta_{0}",1,False,hyperparameter=True)
            if self.q > 0:
                if beta is None:
                    raise ValueError("beta is required if q >= 1")
                
                assert len(beta) == self.q, "weights must have length q"
                for i,ma in enumerate(beta):
                    self.parameters.append(f"beta_{i+1}",float(ma),True,hyperparameter=True)
            for i in range(self.q,self.p-1):
                self.parameters.append(f"beta_{i+1}",float(0.),False,hyperparameter=True)
                
        elif lorentzian_centroids is not None and lorentzian_widths is not None:
             
            assert len(lorentzian_centroids) == len(lorentzian_widths), "lorentzian_centroids and lorentzian_widths must have the same length"
            if self.p % 2 == 0:
                assert jnp.count_nonzero(lorentzian_centroids) == len(lorentzian_centroids), "When p is even, lorentzian_centroids must have non-zero elements"
                assert len(lorentzian_centroids) == self.p//2, "lorentzian_centroids must have p//2 non-zero elements"
            else:
                assert jnp.count_nonzero(lorentzian_centroids)+1 == len(lorentzian_centroids), "When p is odd, lorentzian_centroids must have p//2+1 non-zero elements"
                assert jnp.count_nonzero(lorentzian_centroids) == (self.p-1)//2, "lorentzian_centroids must have p//2+1 non-zero elements"

            roots = lorentzians_to_roots(lorentzian_widths,lorentzian_centroids)
            AR_quad = roots_to_quad(roots)
            for i,ar in enumerate(AR_quad):
                    self.parameters.append(f"a_{i+1}",float(ar),True,hyperparameter=True)
            self.parameters.append("beta_0",float(1.),False,hyperparameter=True)
            
            if self.q == 0:
                assert weights is None, "weights must be None if q = 0"
            else:
                assert len(weights) == self.q, "weights must have length q"
                for i,ma in enumerate(weights):
                        self.parameters.append(f"beta_{i+1}",float(ma),True,hyperparameter=True)
            for i in range(self.q,self.p-1):
                self.parameters.append(f"beta_{i+1}",float(0.),False,hyperparameter=True)
        else:
            raise ValueError("Either AR_roots and MA_roots or AR_quad and MA_quad or lorentzian_centroids, lorentzian_widths and weights must be provided")

    def get_AR_quads(self) -> jax.Array:
        r"""Returns the quadratic coefficients of the AR part of the model.

        Iterates over the parameters of the model and returns the quadratic
        coefficients of the AR part of the model.

        Returns
        -------
        :obj:`jax.Array`
            Quadratic coefficients of the AR part of the model.
        """
        return jnp.array([self.parameters[f"a_{i}"].value for i in range(1,self._p)])

    def get_MA_coeffs(self) -> jax.Array:
        r"""Returns the quadratic coefficients of the AR part of the model.

        Iterates over the parameters of the model and returns the quadratic
        coefficients of the AR part of the model.

        Returns
        -------
        :obj:`jax.Array`
            Quadratic coefficients of the AR part of the model.
        """
        return jnp.array([self.parameters[f"beta_{i}"].value for i in range(self.p)])
    
    def get_AR_roots(self) -> jax.Array:
        r"""Returns the roots of the AR part of the model.
        
        Returns
        -------
        :obj:`jax.Array`
            Roots of the AR part of the model.
        """
        return quad_to_roots(self.get_AR_quads())
    
    def calculate(self, tau: jax.Array) -> jax.Array:
        r"""Compute the autocovariance function of a CARMA(p,q) process."""
        Frac = 0
        roots_AR = self.get_AR_roots()
        beta = self.get_MA_coeffs()
        q = beta.shape[0]
        for k, r in enumerate(roots_AR):
            A, B = 0, 0
            for l in range(q):
                A += beta[l]*r**l
                B += beta[l]*(-r)**l
            Den = -2*jnp.real(r)
            for l, root_AR_bis in enumerate(jnp.delete(roots_AR,k)):
                Den *= (root_AR_bis - r)*(jnp.conjugate(root_AR_bis) + r)
            Frac += A*B/Den*jnp.exp(r*tau)
        return self.parameters["sigma"].value**2 * Frac.real    
