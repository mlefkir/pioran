"""Base representation of a covariance function. It is not meant to be used directly, but rather as a base class to build covariance functions. 
The sum and product of covariance functions are implemented with the ``+`` and ``*`` operators, respectively.
"""
from copy import deepcopy

import equinox as eqx
import jax

from .parameters import ParametersModel
from .utils import EuclideanDistance


class CovarianceFunction(eqx.Module):
    """Represents a covariance function model.

    Bridge between the parameters and the covariance function model. All covariance functions
    inherit from this class.

    Parameters
    ----------
    param_values : :class:`~pioran.parameters.ParametersModel` or  :obj:`list` of :obj:`float`
        Values of the parameters of the covariance function.
    param_names :  :obj:`list` of :obj:`str`
        param_names of the parameters of the covariance function.
    free_parameters :  :obj:`list` of :obj:`bool`
        list` of :obj:`bool` to indicate if the parameters are free or not.

    Raises
    ------
    `TypeError`
        If param_values is not a :obj:`list` of :obj:`float` or a :class:`~pioran.parameters.ParametersModel`.
    """

    parameters: ParametersModel
    """Parameters of the covariance function."""
    expression: str
    """Expression of the covariance function."""

    def __init__(
        self,
        param_values: ParametersModel | list[float],
        param_names: list[str],
        free_parameters: list[bool],
    ):
        """Constructor of the covariance function inherited from the CovarianceFunction class."""
        if isinstance(param_values, ParametersModel):
            self.parameters = param_values
        elif isinstance(param_values, list) or isinstance(param_values, jax.Array):
            self.parameters = ParametersModel(
                param_names=param_names,
                param_values=param_values,
                free_parameters=free_parameters,
            )
        else:
            raise TypeError(
                "The parameters of the covariance function must be a list` of :obj:`float`s or jax.Array or a ParametersModel object."
            )

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

    def get_cov_matrix(self, xq: jax.Array, xp: jax.Array) -> jax.Array:
        """Compute the covariance matrix between two arrays xq, xp.

        The term (xq-xp) is computed using the :func:`~pioran.utils.EuclideanDistance` function from the utils module.

        Parameters
        ----------
        xq : :obj:`jax.Array`
            First array.
        xp : :obj:`jax.Array`
            Second array.

        Returns
        -------
        (N,M) :obj:`jax.Array`
            Covariance matrix.
        """
        dist = EuclideanDistance(xq, xp)
        return self.calculate(dist)

    def __add__(self, other: "CovarianceFunction") -> "SumCovarianceFunction":
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
        other = deepcopy(other)
        other.parameters.increment_IDs(len(self.parameters.values))
        other.parameters.increment_component(max(self.parameters.components))
        return SumCovarianceFunction(self, other)

    def __mul__(self, other: "CovarianceFunction") -> "ProductCovarianceFunction":
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
        other = deepcopy(other)
        other.parameters.increment_IDs(len(self.parameters.values))
        other.parameters.increment_component(max(self.parameters.components))
        return ProductCovarianceFunction(self, other)


class ProductCovarianceFunction(CovarianceFunction):
    """Represents the product of two covariance functions.

    Parameters
    ----------
    cov1 : :obj:`CovarianceFunction`
        First covariance function.
    cov2 : :obj:`CovarianceFunction`
        Second covariance function.
    """

    cov1: CovarianceFunction
    """First covariance function."""
    cov2: CovarianceFunction
    """Second covariance function."""
    parameters: ParametersModel
    """Parameters of the covariance function."""
    expression: str
    """Expression of the total covariance function."""

    def __init__(self, cov1: "CovarianceFunction", cov2: "CovarianceFunction"):
        """Constructor of the ProductCovarianceFunction class.

        Parameters
        ----------
        cov1 : :obj:`CovarianceFunction`
            First covariance function.
        cov2 : :obj:`CovarianceFunction`
            Second covariance function.
        """
        self.cov1 = cov1
        self.cov2 = cov2
        if isinstance(cov1, SumCovarianceFunction) and isinstance(
            cov2, SumCovarianceFunction
        ):
            self.expression = f"({cov1.expression}) * ({cov2.expression})"
        elif isinstance(cov1, SumCovarianceFunction):
            self.expression = f"({cov1.expression}) * {cov2.expression}"
        elif isinstance(cov2, SumCovarianceFunction):
            self.expression = f"{cov1.expression} * ({cov2.expression})"
        else:
            self.expression = f"{cov1.expression} * {cov2.expression}"

        self.parameters = ParametersModel(
            param_names=cov1.parameters.names + cov2.parameters.names,
            param_values=cov1.parameters.values + cov2.parameters.values,
            free_parameters=cov1.parameters.free_parameters
            + cov2.parameters.free_parameters,
            _pars=cov1.parameters._pars + cov2.parameters._pars,
        )

    @eqx.filter_jit
    def calculate(self, x: jax.Array) -> jax.Array:
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
    """Represents the sum of two covariance functions.

    Parameters
    ----------
    cov1 : :obj:`CovarianceFunction`
        First covariance function.
    cov2 : :obj:`CovarianceFunction`
        Second covariance function.
    """

    cov1: CovarianceFunction
    """First covariance function."""
    cov2: CovarianceFunction
    """Second covariance function."""
    parameters: ParametersModel
    """Parameters of the covariance function."""
    expression: str
    """Expression of the total covariance function."""

    def __init__(self, cov1: CovarianceFunction, cov2: CovarianceFunction) -> None:
        """Constructor of the SumCovarianceFunction class."""
        self.cov1 = cov1
        self.cov2 = cov2
        self.expression = f"{cov1.expression} + {cov2.expression}"

        self.parameters = ParametersModel(
            param_names=cov1.parameters.names + cov2.parameters.names,
            param_values=cov1.parameters.values + cov2.parameters.values,
            free_parameters=cov1.parameters.free_parameters
            + cov2.parameters.free_parameters,
            _pars=cov1.parameters._pars + cov2.parameters._pars,
        )

    @eqx.filter_jit
    def calculate(self, x: jax.Array) -> jax.Array:
        """Compute the covariance function at the points x.

        It is the sum of the two covariance functions.

        Parameters
        ----------
        x : :obj:`jax.Array`
            Points where the covariance function is computed.

        Returns
        -------
        :obj:`SumCovarianceFunction`
            Sum of the two covariance functions at the points x.
        """
        return self.cov1.calculate(x) + self.cov2.calculate(x)
