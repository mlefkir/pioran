import equinox as eqx
import jax
import jax.numpy as jnp

from .carma_model import CARMA_model
from .kalman import KalmanFilter
from .tools import sanity_checks


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
    model: CARMA_model
    kalman: KalmanFilter
    
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
        self.model = CARMA_model(p,q,AR_quad=start_AR_quad,beta=start_beta,**kwargs)

        # add a factor to scale the errors
        scale_errors = kwargs.get("scale_errors", True)
        if scale_errors and (observation_errors is not None):
            self.model.parameters.append("nu",1.0,True,hyperparameter=False)

        # add the mean of the observed data as a parameter
        estimate_mean = kwargs.get("estimate_mean", True)
        if estimate_mean:
            self.model.parameters.append("mu",jnp.mean(self.observation_values),True,hyperparameter=False)
        else:
            print("The mean of the training data is not estimated. Be careful of the data included in the training set.")


        # set the kalman filter 
        self.kalman = KalmanFilter(model=self.model,observation_indexes=self.observation_indexes,
                                   observation_values=self.observation_values-self.model.parameters['mu'].value,
                                   observation_errors=self.observation_errors*jnp.sqrt(self.model.parameters['nu'].value))
        
    
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
        s += f"Marginal log likelihood: {self.kalman.log_likelihood():.5f}\n"
        s += self.model.__str__()
        return s

    def __repr__(self) -> str:
        return self.__str__()