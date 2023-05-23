import equinox as eqx
import jax
import jax.numpy as jnp

from .core import GaussianProcess
from .acvf import CARMA_covariance
from .carma_model import CARMA_model
from .kalman import KalmanFilter
from .tools import sanity_checks,reshape_array


class CARMAProcess(eqx.Module):
    """Base class for inference with Continuous autoregressive moving average processes

    Parameters
    ----------
    
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
    S_low: float
    S_high: float
    log_max: float
    log_max_MA: float
    log_min: float
    log_min_MA: float
    prior_sigma: float
    nu_max: float
    nu_min: float
    mu_max: float
    mu_min: float
    
    
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
        self.use_beta = kwargs.get('use_beta',True)
        if self.use_beta and self.q > 1:
            self.model = CARMA_model(p,q,AR_quad=start_AR_quad,beta=start_beta,**kwargs)
        else:
            self.model = CARMA_model(p,q,AR_quad=start_AR_quad,MA_quad=start_beta,**kwargs)
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
    
        # set the prior scales
        self.S_low = kwargs.get("S_low",2)
        self.S_high = kwargs.get("S_high",2)
        self.set_priors_limits()
        
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
    
    def set_priors_limits(self):
        
        # sigma
        std_obs = jnp.std(self.observation_values)
        self.prior_sigma = std_obs * 3
        
        # AR_quad
        f_min = 1/(self.observation_indexes[-1]-self.observation_indexes[0])
        f_max = .5/jnp.min(jnp.diff(self.observation_indexes))
        self.log_min = (f_min / self.S_low)
        self.log_max = (f_max * self.S_high)
        
        # MA_coefs
        self.log_min_MA = -2
        self.log_max_MA = 2
    
        # nu
        self.nu_max = 5
        self.nu_min = 1e-1
        
        # mu
        self.mu_max = 10*jnp.abs(jnp.mean(self.observation_values))
        self.mu_min = -10*jnp.abs(jnp.mean(self.observation_values))
        
        
        return
        
    
    # def priors(self,cube):
    #     params = cube.copy()
        
    #     params[0] = params[0]*self.prior_sigma
    #     for i in range(1,self.p+1):
    #         params[i] = 10**(params[i]*(self.log_max-self.log_min)+self.log_min)
    #     for i in range(self.p+1,self.p+self.q+1):
    #         params[i] = 10**(params[i]*(self.log_max_MA-self.log_min_MA)+self.log_min_MA)
    #     i = self.p+self.q
    #     if self.scale_errors:
    #         params[i+1] = params[i+1]*(self.nu_max-self.nu_min)+self.nu_min
    #     if self.estimate_mean:
    #         params[i+2] = params[i+2]*(self.mu_max-self.mu_min)+self.mu_min
    #     return params    
        
    
    @eqx.filter_jit
    def wrapper_log_marginal_likelihood(self,params) -> float:
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