import jax
import jax.numpy as jnp

from ..acvf import CovarianceFunction
from ..parameters import ParametersModel
from .carma_utils import MA_quad_to_coeff, quad_to_roots, initialise_CARMA_object


class CARMA_covariance(CovarianceFunction):
    r"""Covariance function of a Continuous AutoRegressive Moving Average (CARMA) process.

    Parameters
    ----------
    p : :obj:`int`
        Order of the AR part of the model.
    q : :obj:`int`
        Order of the MA part of the model. 0 <= q < p
    AR_quad : :obj:`list` of :obj:`float`
        Quadratic coefficients of the AR part of the model.
    MA_quad : :obj:`list` of :obj:`float`
        Quadratic coefficients of the MA part of the model.
    beta : :obj:`list` of :obj:`float`
        MA coefficients of the model.
    use_beta : :obj:`bool`
        If True, the MA coefficients are given by the beta parameters. If False, the MA coefficients are given by the quadratic coefficients.
    lorentzian_centroids : :obj:`list` of :obj:`float`
        Centroids of the Lorentzian functions.
    lorentzian_widths : :obj:`list` of :obj:`float`
        Widths of the Lorentzian functions.
    weights : :obj:`list` of :obj:`float`
        Weights of the Lorentzian functions.
    """

    parameters: ParametersModel
    """Parameters of the covariance function."""
    expression: str
    """Expression of the covariance function."""
    p: int
    """Order of the AR part of the model."""
    q: int
    """Order of the MA part of the model. 0 <= q < p"""
    _p: int
    """Order of the AR part of the model. p+1"""
    _q: int
    """Order of the MA part of the model. q+1"""
    use_beta: bool
    """If True, the MA coefficients are given by the beta parameters. If False, the MA coefficients are given by the quadratic coefficients."""

    def __init__(
        self,
        p,
        q,
        AR_quad=None,
        MA_quad=None,
        beta=None,
        use_beta=True,
        lorentzian_centroids=None,
        lorentzian_widths=None,
        weights=None,
        **kwargs,
    ) -> None:
        """Constructor method"""
        sigma = kwargs.get("sigma", 1)

        CovarianceFunction.__init__(
            self, param_values=[sigma], param_names=["sigma"], free_parameters=[True]
        )

        self.p = p
        self.q = q
        assert self.q < self.p, "q must be smaller than p"
        self.expression = f"CARMA({p},{q})"
        self._p = p + 1
        self._q = q + 1
        self.use_beta = use_beta

        initialise_CARMA_object(
            self,
            p,
            q,
            AR_quad,
            MA_quad,
            beta,
            use_beta,
            lorentzian_centroids,
            lorentzian_widths,
            weights,
            **kwargs,
        )

    def get_AR_quads(self) -> jax.Array:
        r"""Returns the quadratic coefficients of the AR part of the model.

        Iterates over the parameters of the model and returns the quadratic
        coefficients of the AR part of the model.

        Returns
        -------
        :obj:`jax.Array`
            Quadratic coefficients of the AR part of the model.
        """
        return jnp.array([self.parameters[f"a_{i}"].value for i in range(1, self._p)])

    def get_MA_quads(self) -> jax.Array:
        """Returns the quadratic coefficients of the MA part of the model.

        Iterates over the parameters of the model and returns the quadratic
        coefficients of the MA part of the model.

        Returns
        -------
        :obj:`jax.Array`
            Quadratic coefficients of the MA part of the model.
        """
        return jnp.array(
            [self.parameters[f"b_{i}"].value for i in range(1, self.q + 1)]
        )

    def get_MA_coeffs(self) -> jax.Array:
        r"""Returns the quadratic coefficients of the AR part of the model.

        Iterates over the parameters of the model and returns the quadratic
        coefficients of the AR part of the model.

        Returns
        -------
        :obj:`jax.Array`
            Quadratic coefficients of the AR part of the model.
        """
        if self.use_beta:
            return jnp.array(
                [self.parameters[f"beta_{i}"].value for i in range(self.p)]
            )
        else:
            return jnp.append(
                MA_quad_to_coeff(self.q, self.get_MA_quads()),
                jnp.zeros(self.p - self.q - 1),
            )

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
                A += beta[l] * r**l
                B += beta[l] * (-r) ** l
            Den = -2 * jnp.real(r)
            for l, root_AR_bis in enumerate(jnp.delete(roots_AR, k)):
                Den *= (root_AR_bis - r) * (jnp.conjugate(root_AR_bis) + r)
            Frac += A * B / Den * jnp.exp(r * tau)
        return self.parameters["sigma"].value ** 2 * Frac.real
