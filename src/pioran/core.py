"""Core class for Gaussian process regression.

"""
from typing import Union

import jax
import equinox as eqx
import jax.numpy as jnp
from jax.scipy.linalg import cholesky, solve, solve_triangular

from .acvf_base import CovarianceFunction
from .psd_base import PowerSpectralDensity
from .psdtoacv import PSDToACV
from .tools import reshape_array, sanity_checks
from .utils.gp_utils import nearest_positive_definite


class GaussianProcess(eqx.Module):
    r""" Class for the Gaussian Process Regression of 1D data. 

    The Gaussian Process Regression is a non-parametric method to estimate the mean and the covariance of a function.

    Parameters
    ----------
    fun : :class:`~pioran.acvf_base.CovarianceFunction` or :class:`~pioran.psd_base.PowerSpectralDensity` 
        Model function associated to the Gaussian Process. Can be a covariance function or a power spectral density.
    observation_indexes : :obj:`jax.Array`
        Indexes of the training data, in this case it is the time.
    observation_values : :obj:`jax.Array`
        Observables of the training data, in this it is flux, count-rate or intensity, etc.
    observation_errors : :obj:`jax.Array`, optional
        Errors on the observables, by default :obj:`None`
    **kwargs : dict
        nb_prediction_points : :obj:`int`, optional
            Number of points to predict, by default 5 * length(training(indexes)).
        prediction_indexes : :obj:`jax.Array`, optional
            Indexes of the prediction data, by default jnp.linspace(jnp.min(observation_indexes),jnp.max(observation_indexes),nb_prediction_points)
        scale_errors : :obj:`bool`, optional
            Scale the errors on the training data by adding a constant, by default True.
        estimate_mean : :obj:`bool`, optional
            Estimate the mean of the training data, by default True.
        S_low : :obj:`float`, optional
            Scaling factor for the lower bound of the PSD, by default 2. See :obj:`PSDToACV` for more details.
        S_high : :obj:`float`, optional
            Scaling factor for the upper bound of the PSD, by default 2. See :obj:`PSDToACV` for more details.

    Methods
    -------
    get_cov(xt, xp, errors=None)
        Compute the covariance matrix between two arrays.
    compute_predictive_distribution(**kwargs)
        Compute the predictive mean and predictive covariance given prediction indexes.
    compute_log_marginal_likelihood()
        Compute the log marginal likelihood.
    wrapper_log_marginal_likelihood(parameters)
        Wrapper to compute the log marginal likelihood.
    wrapper_neg_log_marginal_likelihood(parameters)
        Wrapper to compute the negative log marginal likelihood.
        
    Attributes
    ----------
    model : :class:`~pioran.acvf_base.CovarianceFunction`  or :class:`~pioran.psd_base.PowerSpectralDensity`
        Model associated to the Gaussian Process, can be a covariance function or a power spectral density.
    observation_indexes : :obj:`jax.Array` (n,1)
        Indexes of the training data.
    observation_values : :obj:`jax.Array` of shape (n,1)
        Observabled training data.
    observation_errors : :obj:`jax.Array` of shape (n,1)
        Errors on the training observed data.
    scale_errors : :obj:`bool`
        Scale the errors on the training data by adding a constant, by default True.
    estimate_mean : :obj:`bool`
        Estimate the mean of the training data, by default True.
    analytical_cov : :obj:`bool`
        True if the covariance function is analytical, False if it is estimated from a power spectral density.
    nb_prediction_points : :obj:`int`
        Number of points to predict, by default 5 * length(training(indexes)).

    """
    model: Union[CovarianceFunction,PSDToACV] 
    observation_indexes: jax.Array
    observation_errors: jax.Array
    observation_values: jax.Array
    prediction_indexes: jax.Array
    nb_prediction_points: int
    scale_errors: bool
    estimate_mean: bool
    analytical_cov: bool    
    
    def __init__(self, function: Union[CovarianceFunction,PowerSpectralDensity], observation_indexes, observation_values, observation_errors=None, **kwargs) -> None:
        """Constructor method for the GaussianProcess class.

        """
        # Check if the training arrays have the same shape
        if observation_errors is None:
            sanity_checks(observation_indexes, observation_values)
        else:
            sanity_checks(observation_indexes, observation_values)
            sanity_checks(observation_values, observation_errors)


        if isinstance(function, CovarianceFunction):
            self.analytical_cov = True
            self.model = function

        elif isinstance(function, PowerSpectralDensity):
            self.analytical_cov = False
            S_low = kwargs.get("S_low", 10)
            S_high = kwargs.get("S_high", 10)
            method = kwargs.get("method", "FFT")
            self.model = PSDToACV(function, S_low=S_low, S_high=S_high,T = observation_indexes[-1]-observation_indexes[0],dt =jnp.min(jnp.diff(observation_indexes)),method=method)
        else:
            raise TypeError("The input model must be a CovarianceFunction or a PowerSpectralDensity.")
        
        # add a factor to scale the errors
        self.scale_errors = kwargs.get("scale_errors", True)
        if self.scale_errors and (observation_errors is not None):
            self.model.parameters.append("nu",1.0,True,hyperparameter=False)


        # Reshape the arrays
        self.observation_indexes = reshape_array(observation_indexes)
        self.observation_values = reshape_array(observation_values)
        # add a small number to the errors to avoid singular matrices in the cholesky decomposition
        self.observation_errors = observation_errors.flatten() if observation_errors is not None else jnp.ones_like(self.observation_values)*jnp.sqrt(jnp.finfo(float).eps)

        # add the mean of the observed data as a parameter
        self.estimate_mean = kwargs.get("estimate_mean", True)
        if self.estimate_mean:
            self.model.parameters.append("mu",jnp.mean(self.observation_values),True,hyperparameter=False)
        else:
            print("The mean of the training data is not estimated. Be careful of the data included in the training set.")

        # Prediction of data
        self.nb_prediction_points = kwargs.get("nb_prediction_points", 5*len(self.observation_indexes))
        self.prediction_indexes = kwargs.get('prediction_indexes', reshape_array(jnp.linspace(jnp.min(self.observation_indexes), jnp.max(self.observation_indexes), self.nb_prediction_points)))

    def get_cov(self, xt, xp, errors=None):
        """ Compute the covariance matrix between two arrays. 
        
        To compute the covariance matrix, this function calls the get_cov_matrix method of the model. 
        If the errors are not None, then the covariance matrix is computed for the training dataset, 
        i.e. with observed data as input (xt=xp=training data) and the errors is the "standard deviation".
        The total covariance matrix is computed as:
        
        

        Parameters
        ----------
        xt: array of shape (n,1)
            First array.
        xp: array  of shape (m,1)
            Second array.
        errors: array of shape (n)
            Errors on the observed data, default is None.
            If errors is not None, then the covariance matrix is computed for the training dataset, i.e. with observed 
            data as input (xt=xp=training data) and the errors is the "standard deviation". The total covariance matrix is computed as:

                C = K + nu * sigma^2 * I

                with I the identity matrix, K the covariance matrix, sigma the errors and nu a free parameter in case the errors
                are over or under-estimated.

        Returns
        -------
        array of shape (n,m)
            Covariance matrix between the two arrays.

        """
        # if not errors return the covariance matrix
        if errors is None:
            return self.model.get_cov_matrix(xt, xp)
        # if errors and we want to scale them
        if self.scale_errors:
            return self.model.get_cov_matrix(xt, xp) + self.model.parameters["nu"].value * jnp.diag(errors**2)
        # if we do not want to scale the errors
        return self.model.get_cov_matrix(xt, xp) + jnp.diag(errors**2)

    def get_cov_training(self):
        """ Compute the covariance matrix and other vectors for the training data.

        Returns
        -------
        Cov_xx: array of shape (n,n)
            Covariance matrix for the training data.
        Cov_inv: array of shape (n,n)
            Inverse of Cov_xx.
        alpha: array of shape (n,1)
            alpha = Cov_inv * observation_values (- mu if mu is estimated)
        """

        Cov_xx = self.get_cov(
            self.observation_indexes, self.observation_indexes, errors=self.observation_errors)
        Cov_inv = solve(Cov_xx, jnp.eye(len(self.observation_indexes)))
        if self.estimate_mean:
            alpha = Cov_inv@(self.observation_values -
                             self.model.parameters["mu"].value)
        else:
            alpha = Cov_inv@(self.observation_values)
        return Cov_xx, Cov_inv, alpha

    def compute_predictive_distribution(self, **kwargs):
        """ Compute the predictive mean and the predictive covariance of the GP. 

        The predictive distribution are computed using equations (2.25)  and (2.26) in Rasmussen and Williams (2006)
        Following the notation of the book, x is the training indexes, x* is the predictive indexes, y the training observable, 
        k the covariance function, sig the noise in the observation.
        The predictive distribution is computed as:

        alpha = inv( k(x,x) + sig^2 * I ) * observation_values
        mean = k(x*,x) * alpha
        cov = k(x*,x*) - k(x*,x) * inv( k(x,x) + sig^2 * I ) * k(x,x*)


        Parameters
        ----------
        **kwargs: dict
            prediction_indexes: array of length m, optional
                Indexes of the prediction data, by default jnp.linspace(jnp.min(observation_indexes),jnp.max(observation_indexes),nb_prediction_points)

        Returns
        -------
        predictive_mean: array of shape (m,1)
            Predictive mean of the GP.
        predictive_cov: array of shape (m,m)
            Predictive covariance of the GP.
        """
        # if we want to change the prediction indexes
        if "prediction_indexes" in kwargs:
            prediction_indexes = reshape_array(kwargs["prediction_indexes"])
        else:
            prediction_indexes = self.prediction_indexes
        # Compute the covariance matrix between the training indexes
        _, Cov_inv, alpha = self.get_cov_training()
        # Compute the covariance matrix between the training indexes and the prediction indexes
        Cov_xxp = self.get_cov(self.observation_indexes, prediction_indexes)
        Cov_xpxp = self.get_cov(prediction_indexes,
                                prediction_indexes)

        # Compute the predictive mean
        if self.estimate_mean:
            predictive_mean = Cov_xxp.T@alpha + \
                self.model.parameters["mu"].value
        else:
            predictive_mean = Cov_xxp.T@alpha
        # Compute the predictive covariance and ensure that the covariance matrix is positive definite
        predictive_covariance = nearest_positive_definite(
            Cov_xpxp - Cov_xxp.T@Cov_inv@Cov_xxp)

        return predictive_mean, predictive_covariance
    
    def compute_log_marginal_likelihood(self) -> float:
        r""" Compute the log marginal likelihood of the Gaussian Process.

        The log marginal likelihood is computed using algorithm (2.1) in Rasmussen and Williams (2006)
        Following the notation of the book, :math:`x` are the training indexes, x* is the predictive indexes, y the training observable, 
        k the covariance function, sig the noise in the observation.

        Following Simon's notes simply solve of triangular system instead of inverting the matrix:
        
        :math:`L = {\rm cholesky}( k(x,x) + \nu \sigma^2 \times [I] )`
        
        :math:`z = L^{-1} \times \boldsymbol{y}`
        
        log_marginal_likelihood = - 0.5 * z^T * z - sum(log(diag(L))) - n/2 * log(2*pi)

        Returns
        -------
        log_marginal_likelihood: float
            Log marginal likelihood of the GP.

        """
        Cov_xx = self.get_cov(self.observation_indexes, self.observation_indexes,
                              errors=self.observation_errors)
        # Compute the covariance matrix between the training indexes
        try:
            L = cholesky(Cov_xx, lower=True)
        except:
            L = cholesky(nearest_positive_definite(Cov_xx), lower=True)

        if self.estimate_mean:
            z = solve_triangular(L, self.observation_values-self.model.parameters["mu"].value, lower=True)
        else:
            z = solve_triangular(L, self.observation_values, lower=True)

        return -jnp.take(jnp.sum(jnp.log(jnp.diagonal(L))) + 0.5 * len(self.observation_indexes) * jnp.log(2*jnp.pi) + 0.5 * (z.T@z),0)
    
    @eqx.filter_jit
    def wrapper_log_marginal_likelihood(self, parameters) -> float:
        """ Wrapper to compute the log marginal likelihood in function of the (hyper)parameters. 

        Parameters
        ----------
        parameters: array of shape (n)
            (Hyper)parameters of the covariance function.
            
        Returns
        -------
        float 
            Log marginal likelihood of the GP.
        """
        self.model.parameters.set_free_values(parameters)
        return self.compute_log_marginal_likelihood()
    
    @eqx.filter_jit
    def wrapper_neg_log_marginal_likelihood(self, parameters) -> float:
        """ Wrapper to compute the negative log marginal likelihood in function of the (hyper)parameters. 

        Parameters
        ----------
        parameters: array of shape (n)
            (Hyper)parameters of the covariance function.
            
        Returns
        -------
        float
            Negative log marginal likelihood of the GP.
        """
        self.model.parameters.set_free_values(parameters)
        return -self.compute_log_marginal_likelihood()

    def __str__(self) -> str:
        """String representation of the GP object.
        
        Returns
        -------
        str
            String representation of the GP object.        
        """
        s = 31*"=" +" Gaussian Process "+31*"="+"\n\n"
        s += f"Marginal log likelihood: {self.compute_log_marginal_likelihood():.5f}\n"
        s += self.model.__str__()
        return s
