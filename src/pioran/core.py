"""Gaussian process regression for time series analysis."""
import equinox as eqx
import jax
import jax.numpy as jnp
import tinygp
import celerite2.jax as celerite
from jax.scipy.linalg import cholesky, solve, solve_triangular

from .acvf_base import CovarianceFunction
from .psd_base import PowerSpectralDensity
from .psdtoacv import PSDToACV
from .tools import reshape_array, sanity_checks
from .utils import valid_methods, nearest_positive_definite, tinygp_methods


class GaussianProcess(eqx.Module):
    r"""Gaussian Process Regression of univariate time series.

    Model the time series as a Gaussian Process with a given covariance function or power spectral density. The covariance function
    is given by a :class:`~pioran.acvf_base.CovarianceFunction` object and the power spectral density is given by a :class:`~pioran.psd_base.PowerSpectralDensity` object.

    Parameters
    ----------
    function : :class:`~pioran.acvf_base.CovarianceFunction` or :class:`~pioran.psd_base.PowerSpectralDensity`
        Model function associated to the Gaussian Process. Can be a covariance function or a power spectral density.
    observation_indexes : :obj:`jax.Array`
        Indexes of the observed data, in this case it is the time.
    observation_values : :obj:`jax.Array`
        Observation data.
    observation_errors : :obj:`jax.Array`, optional
        Errors on the observables, by default :obj:`None`
    S_low : :obj:`float`, optional
            Scaling factor to extend the frequency grid to lower values, by default 10. See :obj:`PSDToACV` for more details.
    S_high : :obj:`float`, optional
            Scaling factor to extend the frequency grid to higher values, by default 10. See :obj:`PSDToACV` for more details.
    method : :obj:`str`, optional
        Method to compute the covariance function from the power spectral density, by default 'FFT'.
        Possible values are:
            - 'FFT': use the FFT to compute the autocovariance function.
            - 'NuFFT': use the non-uniform FFT to compute the autocovariance function.
            - 'SHO': approximate the power spectrum as a sum of SHO basis functions to compute the autocovariance function.
    use_tinygp : :obj:`bool`, optional
        Use tinygp to compute the log marginal likelihood, by default False. Should only be used when the power spectrum model
        is expressed as a sum of quasi-separable kernels, i.e. method is not 'FFT' or 'NuFFT'.
    n_components : :obj:`int`, optional
        Number of components to use when using tinygp and the power spectrum model is expressed as a sum of quasi-separable kernels.
    estimate_mean : :obj:`bool`, optional
        Estimate the mean of the observed data, by default True.
    estimate_variance : :obj:`bool`, optional
        Estimate the amplitude of the autocovariance function, by default True.
    scale_errors : :obj:`bool`, optional
        Scale the errors on the observed data by adding a multiplicative factor, by default True.
    log_transform : :obj:`bool`, optional
        Use a log transformation of the data, by default False. This is useful when the data is log-normal distributed.
        Only compatible with the method 'FFT' or 'NuFFT'.
    nb_prediction_points : :obj:`int`, optional
        Number of points to predict, by default 5 * length(observed(indexes)).
    prediction_indexes : :obj:`jax.Array`, optional
        indexes of the prediction data, by default linspace(min(observation_indexes),max(observation_indexes),nb_prediction_points)
    """
    model: CovarianceFunction | PSDToACV
    """Model associated to the Gaussian Process, can be a covariance function or a power spectral density to autocovariance function converter."""
    observation_indexes: jax.Array
    """Indexes of the observed data, in this case it is the time."""
    observation_errors: jax.Array
    """Errors on the observed data."""
    observation_values: jax.Array
    """Observed data."""
    prediction_indexes: jax.Array
    """Indexes of the prediction data."""
    nb_prediction_points: int
    """Number of points to predict, by default 5 * length(observed(indexes))."""
    scale_errors: bool = True
    """Scale the errors on the observed data by adding a multiplicative factor."""
    estimate_mean: bool = True
    """Estimate the mean of the observed data."""
    estimate_variance: bool = False
    """Estimate the amplitude of the autocovariance function."""
    log_transform: bool = False
    """Use a log transformation of the data."""
    use_tinygp: bool = False
    """Use tinygp to compute the log marginal likelihood."""
    propagate_errors: bool = True
    """Propagate the errors on the observed data."""
    use_celerite: bool = False
    def __init__(
        self,
        function: CovarianceFunction | PowerSpectralDensity,
        observation_indexes: jax.Array,
        observation_values: jax.Array,
        observation_errors: jax.Array | None = None,
        S_low: float = 10,
        S_high: float = 10,
        method: str = "FFT",
        use_tinygp: bool = False,
        n_components: int = 0,
        estimate_variance: bool = True,
        estimate_mean: bool = True,
        scale_errors: bool = True,
        log_transform: bool = False,
        nb_prediction_points: int = 0,
        propagate_errors: bool = True,
        prediction_indexes: jax.Array | None = None,
        use_celerite: bool = False,
    ) -> None:
        """Constructor method for the GaussianProcess class."""

        if method not in valid_methods:
            raise ValueError(
                f"Method {method} is not valid. Choose between {valid_methods}"
            )

        if method == "NuFFT":
            try:
                import jax_finufft
            except ImportError:
                raise ImportError(
                    "The NuFFT method requires jax_finufft to be installed."
                )

        self.use_tinygp = use_tinygp
        self.use_celerite = use_celerite
        if method in tinygp_methods and not (use_tinygp or use_celerite):
            raise ValueError(
                f"Method '{method}' can only be used with tinygp, please set `use_tinygp=True`"
            )
        if method not in tinygp_methods and (use_tinygp or use_celerite):
            raise ValueError(f"tinygp is only compatible with method {tinygp_methods}")
        if (self.use_tinygp or self.use_celerite) and (n_components == 0):
            raise ValueError(
                "The number of components must be specified when using tinygp, please set `n_components=...`"
            )

        # Check if the arrays have the same shape
        if observation_errors is None:
            sanity_checks(observation_indexes, observation_values)
        else:
            sanity_checks(observation_indexes, observation_values)
            sanity_checks(observation_values, observation_errors)

        if isinstance(function, CovarianceFunction):
            self.model = function

        elif isinstance(function, PowerSpectralDensity):
            self.estimate_variance = estimate_variance
            self.model = PSDToACV(
                function,
                S_low=S_low,
                S_high=S_high,
                T=observation_indexes[-1] - observation_indexes[0],
                dt=jnp.min(jnp.diff(observation_indexes)),
                method=method,
                n_components=n_components,
                estimate_variance=self.estimate_variance,
                init_variance=jnp.var(observation_values, ddof=1),
                use_celerite = self.use_celerite,
            )
            # self.model.print_info()
        else:
            raise TypeError(
                f"The input model must be a CovarianceFunction or a PowerSpectralDensity, not {type(function)}"
            )
        # add a factor to scale the errors
        self.scale_errors = scale_errors
        if observation_errors is None:
            self.scale_errors = False
        if self.scale_errors and (observation_errors is not None):
            self.model.parameters.append("nu", 1.0, True, hyperparameter=False)

        if not (use_tinygp or use_celerite):
            # Reshape the arrays
            self.observation_indexes = reshape_array(observation_indexes)
            self.observation_values = reshape_array(observation_values)
            # add a small number to the errors to avoid singular matrices in the cholesky decomposition
            self.observation_errors = (
                observation_errors.flatten()
                if observation_errors is not None
                else jnp.ones_like(self.observation_values)
                * jnp.sqrt(jnp.finfo(float).eps)
            )
        else:
            self.observation_indexes = observation_indexes.flatten()
            self.observation_values = observation_values.flatten()
            self.observation_errors = (
                observation_errors.flatten()
                if observation_errors is not None
                else jnp.ones_like(self.observation_values)
                * jnp.sqrt(jnp.finfo(float).eps)
            )
        # add the mean of the observed data as a parameter
        self.estimate_mean = estimate_mean
        self.log_transform = log_transform
        self.propagate_errors = propagate_errors

        if self.estimate_mean:
            self.model.parameters.append(
                "mu",
                jnp.mean(self.observation_values)
                if not self.log_transform
                else jnp.mean(jnp.log(self.observation_values)),
                True,
                hyperparameter=False,
            )
        else:
            print(
                "The mean of the observed data is not estimated. Be careful of the data included in the training set."
            )

        if self.log_transform:
            assert (
                self.estimate_mean
            ), "The mean of the observed data must be estimated to use a log transformation."
            self.model.parameters.append(
                "const",
                0.5 * jnp.min(self.observation_values),
                True,
                hyperparameter=False,
            )

        # Prediction of data
        self.nb_prediction_points = (
            5 * len(self.observation_indexes)
            if nb_prediction_points == 0
            else nb_prediction_points
        )
        self.prediction_indexes = (
            reshape_array(
                jnp.linspace(
                    jnp.min(self.observation_indexes),
                    jnp.max(self.observation_indexes),
                    self.nb_prediction_points,
                )
            )
            if prediction_indexes is None
            else reshape_array(prediction_indexes)
        )

    def get_cov(self, xt:jax.Array, xp:jax.Array, errors: jax.Array|None =None) -> jax.Array:
        r"""Compute the covariance matrix between two arrays.

        To compute the covariance matrix, this function calls the get_cov_matrix method of the model.
        If the errors are not None, then the covariance matrix is computed for the observationst,
        i.e. with observed data as input (xt=xp=observed data) and the errors on the measurement.
        The total covariance matrix is computed as:
        
        .. math:: 
            
            C = K + \nu \sigma ^ 2 \times [I]
        
        With :math:`I` the identity matrix, :math:`K` the covariance matrix, :math:`\sigma` the errors and :math:`\nu` a free parameter to scale the errors.
        

        Parameters
        ----------
        xt: :obj:`jax.Array`
            First array.
        xp: :obj:`jax.Array`
            Second array.
        errors: :obj:`jax.Array`, optional
            Errors on the observed data

        Returns
        -------
        :obj:`jax.Array`
            Covariance matrix between the two arrays.

        """
        # if not errors return the covariance matrix
        if errors is None:
            return self.model.get_cov_matrix(xt, xp)
        # if errors and we want to scale them
        if self.scale_errors:
            return self.model.get_cov_matrix(xt, xp) + self.model.parameters[
                "nu"
            ].value * jnp.diag(errors**2)
        # if we do not want to scale the errors
        return self.model.get_cov_matrix(xt, xp) + jnp.diag(errors**2)

    def get_cov_training(self) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Compute the covariance matrix and other vectors for the observed data.

        Returns
        -------
        :obj:`jax.Array`
            Covariance matrix for the observed data.
        :obj:`jax.Array`
            Inverse of Cov_xx.
        :obj:`jax.Array`
            alpha = Cov_inv * observation_values (- mu if mu is estimated)
        """
        if not self.log_transform:
            Cov_xx = self.get_cov(
                self.observation_indexes,
                self.observation_indexes,
                errors=self.observation_errors,
            )
            Cov_inv = solve(Cov_xx, jnp.eye(len(self.observation_indexes)))
            if self.estimate_mean:
                alpha = Cov_inv @ (
                    self.observation_values - self.model.parameters["mu"].value
                )
            else:
                alpha = Cov_inv @ (self.observation_values)
        else:
            latent_errors = self.observation_errors.flatten() / (
                self.observation_values.flatten() - self.model.parameters["const"].value
            )
            latent_values = (
                jnp.log(
                    jnp.abs(
                        self.observation_values - self.model.parameters["const"].value
                    )
                )
                - self.model.parameters["mu"].value
            )

            Cov_xx = self.get_cov(
                self.observation_indexes, self.observation_indexes, errors=latent_errors
            )
            Cov_inv = solve(Cov_xx, jnp.eye(len(self.observation_indexes)))
            if self.estimate_mean:
                alpha = Cov_inv @ latent_values
            else:
                alpha = Cov_inv @ latent_values
        return Cov_xx, Cov_inv, alpha

    def compute_predictive_distribution(
        self, log_transform:bool |None =None, prediction_indexes: jax.Array | None=None
    ):
        r"""Compute the predictive mean and the predictive covariance of the GP.

        The predictive distribution are computed using equations (2.25)  and (2.26) in Rasmussen and Williams (2006).

        Parameters
        ----------
        log_transform: bool or None, optional
            Predict using a with exponentation of the posterior mean, by default use the default value of the GP.
        prediction_indexes: array of length m, optional
            Indexes of the prediction data, by default jnp.linspace(jnp.min(observation_indexes),jnp.max(observation_indexes),nb_prediction_points)

        Returns
        -------
        :obj:`jax.Array`
            Predictive mean of the GP.
        :obj:`jax.Array`
            Predictive covariance of the GP.
        """
        log_transform = self.log_transform if log_transform is None else log_transform
        # if we want to change the prediction indexes
        if prediction_indexes is not None:
            prediction_indexes = reshape_array(prediction_indexes)
        else:
            prediction_indexes = self.prediction_indexes

        if not self.use_tinygp:
            # Compute the covariance matrix between the observed indexes
            _, Cov_inv, alpha = self.get_cov_training()
            # Compute the covariance matrix between the observed indexes and the prediction indexes
            Cov_xxp = self.get_cov(self.observation_indexes, prediction_indexes)
            Cov_xpxp = self.get_cov(prediction_indexes, prediction_indexes)

            # Compute the predictive mean
            if self.estimate_mean:
                if not log_transform:
                    predictive_mean = (
                        Cov_xxp.T @ alpha + self.model.parameters["mu"].value
                    )
                else:
                    predictive_mean = jnp.exp(
                        Cov_xxp.T @ alpha + self.model.parameters["mu"].value
                    )
            else:
                predictive_mean = Cov_xxp.T @ alpha
            # Compute the predictive covariance and ensure that the covariance matrix is positive definite
            predictive_covariance = nearest_positive_definite(
                Cov_xpxp - Cov_xxp.T @ Cov_inv @ Cov_xxp
            )
        else:
            gp = self.build_gp_tinygp()
            if log_transform:
                y = jnp.log(
                    jnp.abs(
                        self.observation_values - self.model.parameters["const"].value
                    )
                )
            else:
                y = self.observation_values
            predictive_mean, predictive_covariance = gp.predict(
                y, prediction_indexes.flatten(), include_mean=True, return_cov=True
            )

        return predictive_mean, predictive_covariance

    def compute_log_marginal_likelihood_pioran(self) -> float:
        r"""Compute the log marginal likelihood of the Gaussian Process.

        The log marginal likelihood is computed using algorithm (2.1) in Rasmussen and Williams (2006)
        Following the notation of the book, :math:`x` are the observed indexes, x* is the predictive indexes, y the observations,
        k the covariance function, sigma the errors on the observations.

        Solve of triangular system instead of inverting the matrix:

        :math:`L = {\rm cholesky}( k(x,x) + \nu \sigma^2 \times [I] )`

        :math:`z = L^{-1} \times (\boldsymbol{y}-\mu))`

        :math:`\mathcal{L} = - \frac{1}{2} z^T z - \sum_i \log L_{ii} - \frac{n}{2} \log (2 \pi)`

        Returns
        -------
        :obj:`float`
            Log marginal likelihood of the GP.

        """
        if not self.log_transform:
            Cov_xx = self.get_cov(
                self.observation_indexes,
                self.observation_indexes,
                errors=self.observation_errors,
            )
            # Compute the covariance matrix between the observed indexes
            try:
                L = cholesky(Cov_xx, lower=True)
            except:
                L = cholesky(nearest_positive_definite(Cov_xx), lower=True)

            if self.estimate_mean:
                z = solve_triangular(
                    L,
                    self.observation_values - self.model.parameters["mu"].value,
                    lower=True,
                )
            else:
                z = solve_triangular(L, self.observation_values, lower=True)

            return -jnp.take(
                jnp.sum(jnp.log(jnp.diagonal(L)))
                + 0.5 * len(self.observation_indexes) * jnp.log(2 * jnp.pi)
                + 0.5 * (z.T @ z),
                0,
            )

        # if we use a log transformation of the data
        else:
            latent_errors = self.observation_errors.flatten() / (
                self.observation_values.flatten() - self.model.parameters["const"].value
            )
            latent_values = (
                jnp.log(
                    jnp.abs(
                        self.observation_values - self.model.parameters["const"].value
                    )
                )
                - self.model.parameters["mu"].value
            )

            Cov_xx = self.get_cov(
                self.observation_indexes, self.observation_indexes, errors=latent_errors
            )

            try:
                L = cholesky(Cov_xx, lower=True)
            except:
                print("Cholesky failed")
                L = cholesky(nearest_positive_definite(Cov_xx), lower=True)

            z = solve_triangular(L, latent_values, lower=True)
            l = jnp.take(
                jnp.sum(jnp.log(jnp.diagonal(L)))
                + 0.5 * len(self.observation_indexes) * jnp.log(2 * jnp.pi)
                + 0.5 * (z.T @ z),
                0,
            )
            correction = jnp.sum(
                jnp.log(
                    jnp.abs(
                        self.observation_values - self.model.parameters["const"].value
                    )
                )
            )

            return -l + correction
    
    def build_gp_celerite(self):
        r"""Build the Gaussian Process using :obj:`celerite2`.

        This function is called when the power spectrum model is expressed as a sum of quasi-separable kernels.
        In this case, the covariance function is a sum of :obj:`tinygp.kernels.quasisep` objects.

        Returns
        -------
        :obj:`tinygp.GaussianProcess`
            Gaussian Process object.
        """
        x = self.observation_indexes  # time

        # apply log transformation
        if self.log_transform and self.propagate_errors:
            yerr = self.observation_errors.flatten() / (
                self.observation_values.flatten() - self.model.parameters["const"].value
            )
        else:
            yerr = self.observation_errors.flatten()

        if self.estimate_mean and self.scale_errors:
            gp = celerite.GaussianProcess(self.model.ACVF, mean=self.model.parameters["mu"].value)
            gp.compute(x,diag=(self.model.parameters["nu"].value * yerr**2))
            
        elif self.estimate_mean:
            gp = celerite.GaussianProcess(self.model.ACVF, mean=self.model.parameters["mu"].value)
            gp.compute(x)
        elif self.scale_errors:
            gp = celerite.GaussianProcess(self.model.ACVF)
            gp.compute(x,diag=(self.model.parameters["nu"].value * yerr**2))
        else:
            gp = celerite.GaussianProcess(self.model.ACVF)
            gp.compute(x)

        return gp
    
    def build_gp_tinygp(self) -> tinygp.GaussianProcess:
        r"""Build the Gaussian Process using :obj:`tinygp`.

        This function is called when the power spectrum model is expressed as a sum of quasi-separable kernels.
        In this case, the covariance function is a sum of :obj:`tinygp.kernels.quasisep` objects.

        Returns
        -------
        :obj:`tinygp.GaussianProcess`
            Gaussian Process object.
        """
        x = self.observation_indexes  # time

        # apply log transformation
        if self.log_transform and self.propagate_errors:
            yerr = self.observation_errors.flatten() / (
                self.observation_values.flatten() - self.model.parameters["const"].value
            )
        else:
            yerr = self.observation_errors.flatten()

        if self.estimate_mean and self.scale_errors:
            gp = tinygp.GaussianProcess(
                self.model.ACVF,
                x,
                diag=(
                    self.model.parameters["nu"].value * yerr**2
                ),
                mean=self.model.parameters["mu"].value,
            )
        elif self.estimate_mean:
            gp = tinygp.GaussianProcess(
                self.model.ACVF, x, mean=self.model.parameters["mu"].value
            )
        elif self.scale_errors:
            gp = tinygp.GaussianProcess(
                self.model.ACVF,
                x,
                diag=(
                    self.model.parameters["nu"].value * yerr**2
                ),
            )
        else:
            gp = tinygp.GaussianProcess(self.model.ACVF, x)

        return gp
        
    def compute_log_marginal_likelihood_celerite(self) -> jax.Array:
        r"""Compute the log marginal likelihood of the Gaussian Process using celerite.

        This function is called when the power spectrum model is expressed as a sum of quasi-separable kernels.
        In this case, the covariance function is a sum of :obj:`celerite2.jax.Terms` objects.

        Returns
        -------
        :obj:`float`
            Log marginal likelihood of the GP.

        """
        gp = self.build_gp_celerite()

        if self.log_transform:
            y = jnp.log(
                jnp.abs(self.observation_values - self.model.parameters["const"].value)
            )
        else:
            y = self.observation_values

        return gp.log_likelihood(y)


    def compute_log_marginal_likelihood_tinygp(self) -> jax.Array:
        r"""Compute the log marginal likelihood of the Gaussian Process using tinygp.

        This function is called when the power spectrum model is expressed as a sum of quasi-separable kernels.
        In this case, the covariance function is a sum of :obj:`tinygp.kernels.quasisep` objects.

        Returns
        -------
        :obj:`float`
            Log marginal likelihood of the GP.

        """
        gp = self.build_gp_tinygp()

        if self.log_transform:
            y = jnp.log(
                jnp.abs(self.observation_values - self.model.parameters["const"].value)
            )
        else:
            y = self.observation_values

        return gp.log_probability(y)

    def compute_log_marginal_likelihood(self) -> float:
        if self.use_tinygp:
            return self.compute_log_marginal_likelihood_tinygp()
        elif self.use_celerite:
            return self.compute_log_marginal_likelihood_celerite()
        return self.compute_log_marginal_likelihood_pioran()

    def wrapper_log_marginal_likelihood(self, parameters:jax.Array) -> float:
        """Wrapper to compute the log marginal likelihood in function of the (hyper)parameters.

        Parameters
        ----------
        parameters: :obj:`jax.Array`
            (Hyper)parameters of the covariance function.

        Returns
        -------
        :obj:`float`
            Log marginal likelihood of the GP.
        """
        self.model.parameters.set_free_values(parameters)
        return self.compute_log_marginal_likelihood()

    def wrapper_neg_log_marginal_likelihood(self, parameters:jax.Array) -> float:
        """Wrapper to compute the negative log marginal likelihood in function of the (hyper)parameters.

        Parameters
        ----------
        parameters: :obj:`jax.Array` of shape (n)
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
        :obj:`str`
            String representation of the GP object.
        """
        s = 31 * "=" + " Gaussian Process " + 31 * "=" + "\n\n"
        # s += f"Marginal log likelihood: {self.compute_log_marginal_likelihood():.5f}\n"
        s += self.model.__str__()
        return s

    def __repr__(self) -> str:
        return self.__str__()
