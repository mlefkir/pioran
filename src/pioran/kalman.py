import equinox as eqx
import jax
import jax.numpy as jnp

from .carma_model import CARMA_model


class KalmanFilter(eqx.Module):
    r"""Base class for Kalman filters. Inherits from eqx.Module.
    
    Attributes
    ----------
    observation_indexes : :obj:`jax.Array`
        Indexes of the observations, i.e. the times at which the observations are made.
    observation_values : :obj:`jax.Array`
        Values of the observations.
    observation_errors : :obj:`jax.Array`
        Errors of the observations.
    model : :obj:`CARMA_model`
        CARMA model used for the inference.
        
    
    """   
    
    observation_indexes: jax.Array
    observation_values: jax.Array
    observation_errors: jax.Array
    model: CARMA_model
    
    def __init__(self,model,observation_indexes,observation_values,observation_errors):
        """Constructor method

        Parameters
        ----------
        observation_indexes : :obj:`jax.Array`
            Indexes of the observations, i.e. the times at which the observations are made.
        observation_values : :obj:`jax.Array`
            Values of the observations.
        observation_errors : :obj:`jax.Array`
            Errors of the observations.
        model : :obj:`CARMA_model`
            CARMA model used for the inference.
        
        """
        
        self.observation_indexes = observation_indexes
        
        self.model = model
        
        self.model.parameters.append('nu',1,True,hyperparameter=False)
        self.model.parameters.append('mu',1,True,hyperparameter=False)
        
        self.observation_values = observation_values
        self.observation_errors = observation_errors

    
    @eqx.filter_jit
    def Predict(self,X,P,F,Q):
        r"""Predict step of the Kalman filter.
        
        Given the state vector :math:`\boldsymbol{X}_k` and the covariance matrix :math:`\boldsymbol{P}_k` at time :math:`t_k`, 
        this method computes the predicted state vector :math:`\hat{\boldsymbol{X}}_{k+1}` and the predicted covariance matrix 
        :math:`\hat{\boldsymbol{P}}_{k+1}` at time :math:`t_{k+1}`.
        Using the notation of the Kalman filter, this method computes the following equations:
        
        .. math:: :label: kalmanpredict  

            \hat{\boldsymbol{X}}_{k+1} &= {F}_k \boldsymbol{X}_k \\
            \hat{\boldsymbol{P}}_{k+1} &= {F}_k {P}_k {F}_k^\mathrm{T} + {Q}_k
        
        where :math:`{F}_k` is the transition matrix and :math:`{Q}_k` is the covariance matrix of the noise process.
        
        Parameters
        ----------
        X : :obj:`jax.Array`
            State vector.
        P : :obj:`jax.Array`
            Covariance matrix of the state vector.
        F : :obj:`jax.Array`
            Transition matrix.
        Q : :obj:`jax.Array`
            Covariance matrix of the noise process.

        Returns
        -------
        X : :obj:`jax.Array`
            Predicted state vector.
        P : :obj:`jax.Array`
            Covariance matrix of the predicted state vector.
        """
        X = F @ X
        P = F @ P @ jnp.conjugate(F.T) + Q
        return X, P

    @eqx.filter_jit
    def Update(self,X,P,Z,H,R):
        r"""Update step of the Kalman filter.
        
        Given the predicted state vector :math:`\hat{\boldsymbol{X}}_{k+1}` and the predicted 
        covariance matrix :math:`\hat{{P}}_{k+1}` at time :math:`t_{k+1}`, this method computes the
        updated state vector :math:`\boldsymbol{X}_{k+1}`, the updated covariance matrix :math:`{P}_{k+1}`, 
        the measurement residual :math:`\boldsymbol{Y}_{k+1}` and the innovation covariance matrix :math:`{S}_{k+1}` at time :math:`t_{k+1}`.
        
        Using the notation of the Kalman filter, this method computes the following equations:
        
        .. math:: :label: kalmanupdate
        
            \boldsymbol{Y}_{k+1} &= \boldsymbol{Z}_{k+1} - {H}_{k+1} \hat{\boldsymbol{X}}_{k+1} \\
            {S}_{k+1} &= {H}_{k+1} \hat{{P}}_{k+1} {H}_{k+1}^\mathrm{T} + {R}_{k+1} \\
            {K}_{k+1} &= \hat{{P}}_{k+1} {H}_{k+1}^\mathrm{T} {S}_{k+1}^{-1} \\
            \boldsymbol{X}_{k+1} &= \hat{\boldsymbol{X}}_{k+1} + {K}_{k+1} \boldsymbol{Y}_{k+1} \\
            {P}_{k+1} &= ({I} - {K}_{k+1} {H}_{k+1}) \hat{{P}}_{k+1}  

        Parameters
        ----------
        X : :obj:`jax.Array`
            Predicted state vector.
        P : :obj:`jax.Array`
            Covariance matrix of the predicted state vector.
        Z : :obj:`jax.Array`
            Observation vector.
        H : :obj:`jax.Array`
            Observation matrix.
        R : :obj:`jax.Array`
            Covariance matrix of the observation noise.
        
        Returns
        -------
        X : :obj:`jax.Array`
            Updated state vector.
        P : :obj:`jax.Array`
            Covariance matrix of the updated state vector.
        Y : :obj:`jax.Array`
            Measurement residual.
        S : :obj:`jax.Array`
            Innovation covariance matrix.
        """
        
        """    mean = (b_rot@X)
        var = ( b_rot@P@jnp.conj(b_rot.T) + model.parameters['nu'].value * Yerr[i]**2)
        
        # Kalman gain
        K = P@jnp.conj(b_rot.T) / var
        
        
        X = X + K * (y_[i]-(mean+mu))
        P = P - var*K@jnp.conj(K.T)"""
        
        Y = Z - H @ X
        S = R + H @ P @ jnp.conj(H.T)
        K = P @ jnp.conj(H.T) / S
        X += K @ Y
        # P = (jnp.eye(self.model.p) - K@H) @ P
        P -= S * K @ jnp.conj(K.T)
        
        return X,P,Y,S
    
    @eqx.filter_jit
    def one_step_loglike_CAR1(self,carry,xs):
        r"""Compute the log-likelihood of a single observation value. 
        
        This function is used in the :meth:`log_likelihood` method to compute the sequentially the log-likelihood of all the observations values.
        It is called using the :func:`jax.lax.scan` function. This function calls the :meth:`Predict` and :meth:`Update` methods.      
        The one-step log-likelihood is computed using the following equation:
        
        .. math:: :label: onesteploglike
        
            \log p(\boldsymbol{Z}_k|\boldsymbol{Z}_{1:k-1}) = -\frac{1}{2} \log |{S}_k| - \frac{1}{2} \boldsymbol{Y}_k^\mathrm{T} {S}_k^{-1} \boldsymbol{Y}_k 
        
        Parameters
        ----------
        carry : :obj:`tuple`
            Tuple containing the state vector :math:`\boldsymbol{X}_k`, the covariance matrix :math:`\boldsymbol{P}_k` and the log-likelihood :math:`\log p(\boldsymbol{Z}_k|\boldsymbol{Z}_{1:k-1})` at time :math:`t_k`.
        xs : :obj:`tuple`
            Tuple containing the time increment :math:`\Delta t_k`, the observation value :math:`\boldsymbol{Z}_k` and the observation error :math:`\boldsymbol{\epsilon}_k` at time :math:`t_k`.        

        Returns
        -------
        carry : :obj:`tuple`
            Tuple containing the state vector :math:`\boldsymbol{X}_{k+1}`, the covariance matrix :math:`\boldsymbol{P}_{k+1}` and the log-likelihood :math:`\log p(\boldsymbol{Z}_{k+1}|\boldsymbol{Z}_{1:k})` at time :math:`t_{k+1}`.
        xs : :obj:`tuple`
            Tuple containing the time increment :math:`\Delta t_{k+1}`, the observation value :math:`\boldsymbol{Z}_{k+1}` and the observation error :math:`\boldsymbol{\epsilon}_{k+1}` at time :math:`t_{k+1}`.
    
        """
        
        X, P, loglike = carry
        dt, value, error = xs
  
        F, Q, H = self.model.statespace_representation(dt)
        R = error**2
        
        X,P = self.Predict(X,P,F,Q)
        X,P,Y,S = self.Update(X,P,value,H,R)
        
        loglike += -0.5 * jax.lax.cond(self.model.ndims==1,
                         lambda: jnp.log(S) + Y**2/S,
                         lambda: jnp.log(jnp.linalg.det(S)) + Y@jnp.linalg.solve(S,jnp.eye(self.model.p)@Y.T)) -.5*jnp.log(2*jnp.pi)
        
        carry = (X,P,loglike)
        return carry,xs
    
    @eqx.filter_jit
    def one_step_loglike(self,carry,xs):
        r"""Compute the log-likelihood of a single observation value. 
        
        This function is used in the :meth:`log_likelihood` method to compute the sequentially the log-likelihood of all the observations values.
        It is called using the :func:`jax.lax.scan` function. This function calls the :meth:`Predict` and :meth:`Update` methods.      
        The one-step log-likelihood is computed using the following equation:
        
        .. math:: :label: onesteploglike
        
            \log p(\boldsymbol{Z}_k|\boldsymbol{Z}_{1:k-1}) = -\frac{1}{2} \log |{S}_k| - \frac{1}{2} \boldsymbol{Y}_k^\mathrm{T} {S}_k^{-1} \boldsymbol{Y}_k 
        
        Parameters
        ----------
        carry : :obj:`tuple`
            Tuple containing the state vector :math:`\boldsymbol{X}_k`, the covariance matrix :math:`\boldsymbol{P}_k` and the log-likelihood :math:`\log p(\boldsymbol{Z}_k|\boldsymbol{Z}_{1:k-1})` at time :math:`t_k`.
        xs : :obj:`tuple`
            Tuple containing the time increment :math:`\Delta t_k`, the observation value :math:`\boldsymbol{Z}_k` and the observation error :math:`\boldsymbol{\epsilon}_k` at time :math:`t_k`.        

        Returns
        -------
        carry : :obj:`tuple`
            Tuple containing the state vector :math:`\boldsymbol{X}_{k+1}`, the covariance matrix :math:`\boldsymbol{P}_{k+1}` and the log-likelihood :math:`\log p(\boldsymbol{Z}_{k+1}|\boldsymbol{Z}_{1:k})` at time :math:`t_{k+1}`.
        xs : :obj:`tuple`
            Tuple containing the time increment :math:`\Delta t_{k+1}`, the observation value :math:`\boldsymbol{Z}_{k+1}` and the observation error :math:`\boldsymbol{\epsilon}_{k+1}` at time :math:`t_{k+1}`.
    
        """
        
        X, P, V, b_rot, loglike = carry
        dt, value, error = xs
  
        F = self.model.statespace_representation(dt)
        R = error**2

        X,P = self.Predict(X,P-V,F,V)
        X,P,Y,S = self.Update(X,P,value,b_rot,R)
        
        loglike -= 0.5 * jax.lax.cond(self.model.ndims==1,
            lambda: jnp.log(S) + Y**2/S,
            lambda: jnp.log(jnp.linalg.det(S)) + Y@jnp.linalg.solve(S,jnp.eye(self.model.ndims)@Y.T)) +.5*jnp.log(2*jnp.pi)
                
        carry = (X, P, V, b_rot, loglike)
        return carry, xs
    
    @eqx.filter_jit
    def log_likelihood(self) -> float:
        
        
        dt = jnp.insert(jnp.diff(self.observation_indexes),0,0.)
        xs = ([dt,
               self.observation_values-self.model.parameters['mu'].value,
               jnp.sqrt(self.model.parameters['nu'].value)*self.observation_errors])

        if self.model.p == 1:
            X, P = self.model.init_statespace()
            loglike = jnp.zeros((1,1))
            carry = (X,P,loglike)
            D ,_ = jax.lax.scan(self.one_step_loglike_CAR1,carry,xs)
            return jnp.take(D[2],0) 
        
        X, P, V, b_rot, loglike = self.model.init_statespace(y_0=self.observation_values[0]-self.model.parameters['mu'].value,
                                    errsize=self.observation_errors[0]*jnp.sqrt(self.model.parameters['nu'].value))
                            
        carry = (X, P, V, b_rot, loglike)
        xs = ([jnp.diff(self.observation_indexes),
               self.observation_values[1:]-self.model.parameters['mu'].value,
               jnp.sqrt(self.model.parameters['nu'].value)*self.observation_errors[1:]])
        D ,_ = jax.lax.scan(self.one_step_loglike,carry,xs)
        
        return jnp.take(D[4],0) 
    
    @eqx.filter_jit
    def wrapper_loglike(self,params) -> float:
        self.model.parameters.set_free_values(params)
        return self.log_likelihood().real
