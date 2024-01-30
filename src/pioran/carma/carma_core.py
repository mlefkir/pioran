import equinox as eqx
import jax
import jax.numpy as jnp

from ..core import GaussianProcess
from .carma_acvf import CARMA_covariance
from .carma_model import CARMA_model
from ..tools import sanity_checks,reshape_array
from .kalman import KalmanFilter


class CARMAProcess(eqx.Module):
    """Base class for inference with Continuous autoregressive moving average processes

    Parameters
    ----------
    p : :obj:`int`
        Order of the AR polynomial.
    q : :obj:`int`
        Order of the MA polynomial.
    observation_indexes : :obj:`jax.Array`
        Indexes of the observations.
    observation_values : :obj:`jax.Array`
        Values of the observations.
    observation_errors : :obj:`jax.Array`
        Errors of the observations, if None, the errors are set to sqrt(eps).
    kwargs : :obj:`dict`
        Additional arguments to pass to the CARMA model.
        AR_quad : :obj:`jax.Array` Quadratic coefficients of the AR polynomial.
        beta : :obj:`jax.Array` Coefficients of the MA polynomial.
        use_beta : :obj:`bool` If True, uses the beta coefficients otherwise uses the quadratic coefficients of the MA polynomial.
        scale_errors : :obj:`bool` If True, scales the errors by a factor nu.
        estimate_mean : :obj:`bool` If True, estimates the mean of the process.

    Attributes
    ----------
    p : :obj:`int`
        Order of the AR polynomial.
    q : :obj:`int`
        Order of the MA polynomial.
    observation_indexes : :obj:`jax.Array`
        Indexes of the observations.
    observation_values : :obj:`jax.Array`
        Values of the observations.
    observation_errors : :obj:`jax.Array`
        Errors of the observations, if None, the errors are set to sqrt(eps).
    prediction_indexes : :obj:`jax.Array`
        Indexes of the predictions.
    model : :obj:`CARMA_model`
        CARMA model.
    kalman : :obj:`KalmanFilter`
        Kalman filter associated to the CARMA model.
    use_beta : :obj:`bool`
        If True, uses the beta coefficients otherwise uses the quadratic coefficients of the MA polynomial.
    scale_errors : :obj:`bool`
        If True, scales the errors by a factor nu.
    estimate_mean : :obj:`bool`
        If True, estimates the mean of the process.
    nb_prediction_points : :obj:`int`
        Number of prediction points.   
    
    """
    p: int
    q: int
    observation_indexes: jax.Array
    observation_values: jax.Array
    observation_errors: jax.Array
    prediction_indexes: jax.Array
    model: CARMA_model
    kalman: KalmanFilter
    use_beta: bool
    estimate_mean: bool
    scale_errors: bool
    nb_prediction_points: int 
    
    def __init__(self,p: int,q: int,observation_indexes: jax.Array,observation_values: jax.Array,observation_errors=None,**kwargs) -> None:
        
        if observation_errors is None:
            sanity_checks(observation_indexes, observation_values)
        else:
            sanity_checks(observation_indexes, observation_values)
            sanity_checks(observation_values, observation_errors)


        self.p = p
        self.q = q
                    
        assert self.q < self.p, "q must be smaller than p"
        
        self.observation_indexes = observation_indexes
        self.observation_values = observation_values
        self.observation_errors = observation_errors if observation_errors is not None else jnp.ones_like(self.observation_errors)*jnp.sqrt(jnp.finfo(float).eps)

        # set the model
        start_AR_quad = kwargs.get('AR_quad',jnp.ones(self.p))
        start_beta = kwargs.get('beta',jnp.ones(self.q) if self.q > 0 else None)
        self.use_beta = kwargs.pop('use_beta',True)
        if self.use_beta and self.q >= 1:
            self.model = CARMA_model(p,q,AR_quad=start_AR_quad,beta=start_beta,use_beta=self.use_beta,**kwargs)
        else:
            self.model = CARMA_model(p,q,AR_quad=start_AR_quad,MA_quad=start_beta,use_beta=self.use_beta,**kwargs)
        # add a factor to scale the errors
        self.scale_errors = kwargs.get("scale_errors", True)
        if self.scale_errors and (observation_errors is not None):
            self.model.parameters.append("nu",1.0,True,hyperparameter=False)
        else:
            self.model.parameters.append("nu",1.0,False,hyperparameter=False)

        # add the mean of the observed data as a parameter
        self.estimate_mean = kwargs.get("estimate_mean", True)
        if self.estimate_mean:
            self.model.parameters.append("mu",jnp.mean(self.observation_values),True,hyperparameter=False)
        else:
            print("The mean of the training data is not estimated. Be careful of the data used.")
            self.model.parameters.append("mu",0,False,hyperparameter=False)

        # set the kalman filter 
        self.kalman = KalmanFilter(model=self.model,
                                   observation_indexes=self.observation_indexes,
                                   observation_values=self.observation_values,
                                   observation_errors=self.observation_errors)
    
        
        self.nb_prediction_points = kwargs.get("nb_prediction_points", 5*len(self.observation_indexes))
        self.prediction_indexes = kwargs.get('prediction_indexes', reshape_array(jnp.linspace(jnp.min(self.observation_indexes), jnp.max(self.observation_indexes), self.nb_prediction_points)))

        
        
    def compute_predictive_distribution(self,**kwargs):
        if self.use_beta:
            acvf = CARMA_covariance(p=self.model.p,q=self.model.q,
                                    AR_quad=self.model.get_AR_quads(),
                                    beta = self.model.get_MA_coeffs()[1:] if self.model.q > 0 else None,use_beta=True)
        else:
            acvf = CARMA_covariance(p=self.model.p,q=self.model.q,
                                    AR_quad=self.model.get_AR_quads(),
                                    MA_quad = self.model.get_MA_quads() if self.model.q > 0 else None,use_beta=False)
        gp = GaussianProcess(function=acvf,observation_indexes=self.observation_indexes,
                             observation_values=self.observation_values,observation_errors=self.observation_errors,
                             estimate_mean=self.estimate_mean,scale_errors=self.scale_errors)
        gp.model.parameters.set_free_values(self.model.parameters.free_values)
        
        return gp.compute_predictive_distribution(**kwargs)
        
    def compute_log_marginal_likelihood(self) -> float:
        return self.kalman.log_likelihood()
        
    @eqx.filter_jit
    def wrapper_log_marginal_likelihood(self,params) -> float:
        """ Wrapper to compute the log marginal likelihood in function of the (hyper)parameters. 

        Parameters
        ----------
        parameters: array of shape (n)
            (Hyper)parameters of the process.
            
        Returns
        -------
        float 
            Log marginal likelihood of the CARMA process.
        """
        self.model.parameters.set_free_values(params)
        return self.kalman.log_likelihood()

    def __str__(self) -> str:
        """String representation of the CARMA object.
        
        Returns
        -------
        str
            String representation of the CARMA object.        
        """
        s = 31*"=" +" CARMA Process "+31*"="+"\n\n"
        # s += f"Marginal log likelihood: {self.kalman.log_likelihood():.5f}\n"
        s += self.model.__str__()
        return s

    def __repr__(self) -> str:
        return self.__str__()