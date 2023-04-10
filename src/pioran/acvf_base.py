"""Module for classes representing the covariance function.
"""

import jax.numpy as jnp
import equinox as eqx

from .parameters import ParametersModel
from .utils import EuclideanDistance


class CovarianceFunction(eqx.Module):
    """Master class for covariance functions, inherited from the :class:`equinox.Module` class.

    Bridge between the parameters and the covariance function. The covariance functions
    inherit from this class.
    
    Parameters
    ----------
    param_values : :class:`~pioran.parameters.ParametersModel` or  :obj:`list of float`
        Values of the parameters of the covariance function.
    param_names :  :obj:`list of str`
        param_names of the parameters of the covariance function.
    free_parameters :  :obj:`list of bool`
        List of bool to indicate if the parameters are free or not.

    Raises
    ------
    `TypeError`
        If param_values is not a `list of float` or a :class:`~pioran.parameters.ParametersModel`.

    Attributes
    ----------
    parameters : :class:`~pioran.parameters.ParametersModel`
        Parameters of the covariance function.
    expression : :obj:`str`
        Expression of the covariance function.

    Methods
    -------
    __init__(param_values, param_names, free_parameters)
        Constructor of the covariance function class.
    __str__()
        String representation of the covariance function.
    __add__(other)
        Overload the + operator to add two covariance functions.
    __mul__(other)
        Overload the * operator to multiply two covariance functions.
    get_cov_matrix(xp,xq)
        Returns the covariance matrix.

    """    
    parameters: ParametersModel
    expression: str
    
    def __init__(self, param_values,param_names, free_parameters):
        """Constructor of the covariance function inherited from the CovarianceFunction class.
        """ 
        if isinstance(param_values, ParametersModel):
            self.parameters = param_values
        elif isinstance(param_values, list) or isinstance(param_values, jnp.ndarray):
            self.parameters = ParametersModel( param_names=param_names, param_values=param_values, free_parameters=free_parameters)
        else:
            raise TypeError(
                "The parameters of the covariance function must be a list of floats or jnp.ndarray or a ParametersModel object.")

    def __str__(self) -> str:  # pragma: no cover
        """String representation of the covariance function.

        Returns
        -------
        :obj:`str`
            String representation of the covariance function.
            Include the representation of the parameters.
        """    
        s = f"Covariance function: {self.expression}\n"
        s += f"Number of parameters: {len(self.parameters.values)}\n"
        s += self.parameters.__str__()
        return s
    
    def __repr__(self) -> str:  # pragma: no cover
        """Representation of the covariance function.

        Returns
        -------
        :obj:`str`
            Representation of the covariance function.
            Include the representation of the parameters.
        """    
        return self.__str__()
    
    def get_cov_matrix(self, xq, xp) -> jnp.ndarray:
        """Compute the covariance matrix between two arrays xq, xp.

        The term (xq-xp) is computed using the :func:`~pioran.utils.EuclideanDistance` function from the utils module.

        Parameters
        ----------
        xq : (N,1) :obj:`jax.Array`
            First array.
        xp : (M,1) :obj:`jax.Array`
            Second array.

        Returns
        -------
        (N,M) :obj:`jax.Array`
            Covariance matrix.
        """
        # Compute the Euclidean distance between the query and the points
        dist = EuclideanDistance(xq, xp)
        # Compute the covariance matrix        
        return self.calculate(dist)
    
    def __add__(self, other) -> "SumCovarianceFunction":
        """Overload of the + operator to add two covariance functions.

        Parameters
        ----------
        other : :obj:`CovarianceFunction`
            Covariance function to add.

        Returns
        -------
        :obj:`SumCovarianceFunction`
            Sum of the two covariance functions.
        """
        other.parameters.increment_IDs(len(self.parameters.values))
        other.parameters.increment_component(max(self.parameters.components))
        return SumCovarianceFunction(self, other)
    
    def __mul__(self, other) -> "ProductCovarianceFunction":
        """Overload of the * operator to multiply two covariance functions.
        
        Parameters
        ----------
        other : :obj:`CovarianceFunction`
            Covariance function to multiply.
        
        Returns
        -------
        :obj:`ProductCovarianceFunction`
            Product of the two covariance functions.
        """
        
        other.parameters.increment_IDs(len(self.parameters.values))
        other.parameters.increment_component(max(self.parameters.components))
        return ProductCovarianceFunction(self, other)
    

class ProductCovarianceFunction(CovarianceFunction):
    """Class for the product of two covariance functions.

    Parameters
    ----------
    cov1 : :obj:`CovarianceFunction`
        First covariance function.
    cov2 : :obj:`CovarianceFunction`
        Second covariance function.

    Attributes
    ----------
    cov1 : :obj:`CovarianceFunction`
        First covariance function.
    cov2 : :obj:`CovarianceFunction`
        Second covariance function.
    parameters : :class:`~pioran.parameters.ParametersModel`
        Parameters of the covariance function.
    expression : `str`
        Expression of the total covariance function.

    Methods
    -------
    calculate(x)
        Compute the product of the two covariance functions.
    """
    cov1: CovarianceFunction
    cov2: CovarianceFunction
    parameters: ParametersModel
    expression: str
    
    def __init__(self, cov1, cov2):
        """Constructor of the SumCovarianceFunction class."""
        self.cov1 = cov1
        self.cov2 = cov2
        if isinstance(cov1, SumCovarianceFunction) and isinstance(cov2, SumCovarianceFunction):
            self.expression = f'({cov1.expression}) * ({cov2.expression})'
        elif isinstance(cov1, SumCovarianceFunction):
            self.expression = f'({cov1.expression}) * {cov2.expression}'
        elif isinstance(cov2, SumCovarianceFunction):
            self.expression = f'{cov1.expression} * ({cov2.expression})'
        else:
            self.expression = f'{cov1.expression} * {cov2.expression}'     
        
        
        self.parameters = ParametersModel(param_names=cov1.parameters.names + cov2.parameters.names,
                                          param_values=cov1.parameters.values + cov2.parameters.values,
                                          free_parameters=cov1.parameters.free_parameters + cov2.parameters.free_parameters,
                                          _pars=cov1.parameters._pars + cov2.parameters._pars)
    
    @eqx.filter_jit
    def calculate(self, x) -> jnp.ndarray:
        """Compute the covariance function at the points x.
        
        It is the product of the two covariance functions.
        
        Parameters
        ----------
        x : :obj:`jax.Array` 
            Points where the covariance function is computed.
        
        Returns
        -------
        Product of the two covariance functions at the points x.
        
        """
        return self.cov1.calculate(x) * self.cov2.calculate(x)
    
class SumCovarianceFunction(CovarianceFunction):
    """Class for the sum of two covariance functions.

    Parameters
    ----------
    cov1 : :obj:`CovarianceFunction`
        First covariance function.
    cov2 : :obj:`CovarianceFunction`
        Second covariance function.

    Attributes
    ----------
    cov1 : :obj:`CovarianceFunction`
        First covariance function.
    cov2 : :obj:`CovarianceFunction`
        Second covariance function.
    parameters : :class:`~pioran.parameters.ParametersModel`
        Parameters of the covariance function.
    expression : `str`
        Expression of the total covariance function.

    Methods
    -------
    calculate(x)
        Compute the sum of the two covariance functions.
    """
    cov1: CovarianceFunction
    cov2: CovarianceFunction
    parameters: ParametersModel
    expression: str
    
    def __init__(self, cov1, cov2):
        """Constructor of the SumCovarianceFunction class."""
        self.cov1 = cov1
        self.cov2 = cov2
        self.expression = f'{cov1.expression} + {cov2.expression}'
        
        self.parameters = ParametersModel(param_names=cov1.parameters.names + cov2.parameters.names,
                                          param_values=cov1.parameters.values + cov2.parameters.values,
                                          free_parameters=cov1.parameters.free_parameters + cov2.parameters.free_parameters,
                                          _pars=cov1.parameters._pars + cov2.parameters._pars)
    
    @eqx.filter_jit
    def calculate(self, x) -> jnp.ndarray:
        """Compute the covariance function at the points x.
        
        It is the sum of the two covariance functions.
        
        Parameters
        ----------
        x : :obj:`jax.Array` 
            Points where the covariance function is computed.
        
        Returns
        -------
        Sum of the two covariance functions at the points x.
        
        """
        return self.cov1.calculate(x) + self.cov2.calculate(x)
