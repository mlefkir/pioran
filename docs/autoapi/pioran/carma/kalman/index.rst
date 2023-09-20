
kalman
======

.. py:module:: pioran.carma.kalman


Overview
--------

.. list-table:: Classes
   :header-rows: 0
   :widths: auto
   :class: summarytable

   * - :py:obj:`KalmanFilter <pioran.carma.kalman.KalmanFilter>`
     - Base class for Kalman filters. Inherits from eqx.Module.




Classes
-------

.. py:class:: KalmanFilter(model, observation_indexes, observation_values, observation_errors)

   Bases: :py:obj:`equinox.Module`

   
   Base class for Kalman filters. Inherits from eqx.Module.














   :Attributes:

       **observation_indexes** : :obj:`jax.Array`
           Indexes of the observations, i.e. the times at which the observations are made.

       **observation_values** : :obj:`jax.Array`
           Values of the observations.

       **observation_errors** : :obj:`jax.Array`
           Errors of the observations.

       **model** : :obj:`CARMA_model`
           CARMA model used for the inference.


   ..
       !! processed by numpydoc !!

   .. rubric:: Overview

   .. list-table:: Attributes
      :header-rows: 0
      :widths: auto
      :class: summarytable

      * - :py:obj:`observation_indexes <pioran.carma.kalman.KalmanFilter.observation_indexes>`
        - \-
      * - :py:obj:`observation_values <pioran.carma.kalman.KalmanFilter.observation_values>`
        - \-
      * - :py:obj:`observation_errors <pioran.carma.kalman.KalmanFilter.observation_errors>`
        - \-
      * - :py:obj:`model <pioran.carma.kalman.KalmanFilter.model>`
        - \-


   .. list-table:: Methods
      :header-rows: 0
      :widths: auto
      :class: summarytable

      * - :py:obj:`Predict <pioran.carma.kalman.KalmanFilter.Predict>`\ (X, P, F, Q)
        - Predict step of the Kalman filter.
      * - :py:obj:`Update <pioran.carma.kalman.KalmanFilter.Update>`\ (X, P, Z, H, R)
        - Update step of the Kalman filter.
      * - :py:obj:`one_step_loglike_CAR1 <pioran.carma.kalman.KalmanFilter.one_step_loglike_CAR1>`\ (carry, xs)
        - Compute the log-likelihood of a single observation value.
      * - :py:obj:`one_step_loglike_CARMA <pioran.carma.kalman.KalmanFilter.one_step_loglike_CARMA>`\ (carry, xs)
        - Compute the log-likelihood of a single observation value.
      * - :py:obj:`log_likelihood <pioran.carma.kalman.KalmanFilter.log_likelihood>`\ ()
        - \-
      * - :py:obj:`wrapper_log_marginal_likelihood <pioran.carma.kalman.KalmanFilter.wrapper_log_marginal_likelihood>`\ (params)
        - \-


   .. rubric:: Members

   .. py:attribute:: observation_indexes
      :type: jax.Array

      

   .. py:attribute:: observation_values
      :type: jax.Array

      

   .. py:attribute:: observation_errors
      :type: jax.Array

      

   .. py:attribute:: model
      :type: pioran.carma.carma_model.CARMA_model

      

   .. py:method:: Predict(X, P, F, Q)

      
      Predict step of the Kalman filter.

      Given the state vector :math:`\boldsymbol{X}_k` and the covariance matrix :math:`\boldsymbol{P}_k` at time :math:`t_k`, 
      this method computes the predicted state vector :math:`\hat{\boldsymbol{X}}_{k+1}` and the predicted covariance matrix 
      :math:`\hat{\boldsymbol{P}}_{k+1}` at time :math:`t_{k+1}`.
      Using the notation of the Kalman filter, this method computes the following equations:

      .. math:: :label: kalmanpredict  

          \hat{\boldsymbol{X}}_{k+1} &= {F}_k \boldsymbol{X}_k \\
          \hat{\boldsymbol{P}}_{k+1} &= {F}_k {P}_k {F}_k^\mathrm{T} + {Q}_k

      where :math:`{F}_k` is the transition matrix and :math:`{Q}_k` is the covariance matrix of the noise process.

      :Parameters:

          **X** : :obj:`jax.Array`
              State vector.

          **P** : :obj:`jax.Array`
              Covariance matrix of the state vector.

          **F** : :obj:`jax.Array`
              Transition matrix.

          **Q** : :obj:`jax.Array`
              Covariance matrix of the noise process.

      :Returns:

          **X** : :obj:`jax.Array`
              Predicted state vector.

          **P** : :obj:`jax.Array`
              Covariance matrix of the predicted state vector.













      ..
          !! processed by numpydoc !!

   .. py:method:: Update(X, P, Z, H, R)

      
      Update step of the Kalman filter.

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

      :Parameters:

          **X** : :obj:`jax.Array`
              Predicted state vector.

          **P** : :obj:`jax.Array`
              Covariance matrix of the predicted state vector.

          **Z** : :obj:`jax.Array`
              Observation vector.

          **H** : :obj:`jax.Array`
              Observation matrix.

          **R** : :obj:`jax.Array`
              Covariance matrix of the observation noise.

      :Returns:

          **X** : :obj:`jax.Array`
              Updated state vector.

          **P** : :obj:`jax.Array`
              Covariance matrix of the updated state vector.

          **Y** : :obj:`jax.Array`
              Measurement residual.

          **S** : :obj:`jax.Array`
              Innovation covariance matrix.













      ..
          !! processed by numpydoc !!

   .. py:method:: one_step_loglike_CAR1(carry, xs)

      
      Compute the log-likelihood of a single observation value. 

      This function is used in the :meth:`log_likelihood` method to compute the sequentially the log-likelihood of all the observations values.
      It is called using the :func:`jax.lax.scan` function. This function calls the :meth:`Predict` and :meth:`Update` methods.      
      The one-step log-likelihood is computed using the following equation:

      .. math:: :label: onesteploglike

          \log p(\boldsymbol{Z}_k|\boldsymbol{Z}_{1:k-1}) = -\frac{1}{2} \log |{S}_k| - \frac{1}{2} \boldsymbol{Y}_k^\mathrm{T} {S}_k^{-1} \boldsymbol{Y}_k 

      :Parameters:

          **carry** : :obj:`tuple`
              Tuple containing the state vector :math:`\boldsymbol{X}_k`, the covariance matrix :math:`\boldsymbol{P}_k` and the log-likelihood :math:`\log p(\boldsymbol{Z}_k|\boldsymbol{Z}_{1:k-1})` at time :math:`t_k`.

          **xs** : :obj:`tuple`
              Tuple containing the time increment :math:`\Delta t_k`, the observation value :math:`\boldsymbol{Z}_k` and the observation error :math:`\boldsymbol{\epsilon}_k` at time :math:`t_k`.        

      :Returns:

          **carry** : :obj:`tuple`
              Tuple containing the state vector :math:`\boldsymbol{X}_{k+1}`, the covariance matrix :math:`\boldsymbol{P}_{k+1}` and the log-likelihood :math:`\log p(\boldsymbol{Z}_{k+1}|\boldsymbol{Z}_{1:k})` at time :math:`t_{k+1}`.

          **xs** : :obj:`tuple`
              Tuple containing the time increment :math:`\Delta t_{k+1}`, the observation value :math:`\boldsymbol{Z}_{k+1}` and the observation error :math:`\boldsymbol{\epsilon}_{k+1}` at time :math:`t_{k+1}`.













      ..
          !! processed by numpydoc !!

   .. py:method:: one_step_loglike_CARMA(carry, xs)

      
      Compute the log-likelihood of a single observation value. 

      This function is used in the :meth:`log_likelihood` method to compute the sequentially the log-likelihood of all the observations values.
      It is called using the :func:`jax.lax.scan` function. This function calls the :meth:`Predict` and :meth:`Update` methods.      
      The one-step log-likelihood is computed using the following equation:

      .. math:: :label: onesteploglike

          \log p(\boldsymbol{Z}_k|\boldsymbol{Z}_{1:k-1}) = -\frac{1}{2} \log |{S}_k| - \frac{1}{2} \boldsymbol{Y}_k^\mathrm{T} {S}_k^{-1} \boldsymbol{Y}_k 

      :Parameters:

          **carry** : :obj:`tuple`
              Tuple containing the state vector :math:`\boldsymbol{X}_k`, the covariance matrix :math:`\boldsymbol{P}_k` and the log-likelihood :math:`\log p(\boldsymbol{Z}_k|\boldsymbol{Z}_{1:k-1})` at time :math:`t_k`.

          **xs** : :obj:`tuple`
              Tuple containing the time increment :math:`\Delta t_k`, the observation value :math:`\boldsymbol{Z}_k` and the observation error :math:`\boldsymbol{\epsilon}_k` at time :math:`t_k`.        

      :Returns:

          **carry** : :obj:`tuple`
              Tuple containing the state vector :math:`\boldsymbol{X}_{k+1}`, the covariance matrix :math:`\boldsymbol{P}_{k+1}` and the log-likelihood :math:`\log p(\boldsymbol{Z}_{k+1}|\boldsymbol{Z}_{1:k})` at time :math:`t_{k+1}`.

          **xs** : :obj:`tuple`
              Tuple containing the time increment :math:`\Delta t_{k+1}`, the observation value :math:`\boldsymbol{Z}_{k+1}` and the observation error :math:`\boldsymbol{\epsilon}_{k+1}` at time :math:`t_{k+1}`.













      ..
          !! processed by numpydoc !!

   .. py:method:: log_likelihood() -> float


   .. py:method:: wrapper_log_marginal_likelihood(params) -> float







