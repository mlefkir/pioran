
carma
=====

.. py:module:: pioran.carma

.. autoapi-nested-parse::

   
   CARMA process module for Python
















   ..
       !! processed by numpydoc !!


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   carma_acvf/index.rst
   carma_core/index.rst
   carma_model/index.rst
   carma_utils/index.rst
   kalman/index.rst


Overview
--------

.. list-table:: Classes
   :header-rows: 0
   :widths: auto
   :class: summarytable

   * - :py:obj:`CARMAProcess <pioran.carma.CARMAProcess>`
     - Base class for inference with Continuous autoregressive moving average processes
   * - :py:obj:`CARMA_model <pioran.carma.CARMA_model>`
     - Base class for Continuous-time AutoRegressive Moving Average (CARMA) models. Inherits from eqxinox.Module.
   * - :py:obj:`CARMA_covariance <pioran.carma.CARMA_covariance>`
     - Covariance function of a Continuous AutoRegressive Moving Average (CARMA) process.
   * - :py:obj:`KalmanFilter <pioran.carma.KalmanFilter>`
     - Base class for Kalman filters. Inherits from eqx.Module.




Classes
-------

.. py:class:: CARMAProcess(p: int, q: int, observation_indexes: jax.Array, observation_values: jax.Array, observation_errors=None, **kwargs)

   Bases: :py:obj:`equinox.Module`

   
   Base class for inference with Continuous autoregressive moving average processes


   :Parameters:

       **p** : :obj:`int`
           Order of the AR polynomial.

       **q** : :obj:`int`
           Order of the MA polynomial.

       **observation_indexes** : :obj:`jax.Array`
           Indexes of the observations.

       **observation_values** : :obj:`jax.Array`
           Values of the observations.

       **observation_errors** : :obj:`jax.Array`
           Errors of the observations, if None, the errors are set to sqrt(eps).

       **kwargs** : :obj:`dict`
           Additional arguments to pass to the CARMA model.
           - AR_quad : :obj:`jax.Array`
               Quadratic coefficients of the AR polynomial.
           - beta : :obj:`jax.Array`
               Coefficients of the MA polynomial.
           - use_beta : :obj:`bool`
               If True, uses the beta coefficients otherwise uses the quadratic coefficients of the MA polynomial.
           - scale_errors : :obj:`bool`
               If True, scales the errors by a factor nu.
           - estimate_mean : :obj:`bool`
               If True, estimates the mean of the process.












   :Attributes:

       **p** : :obj:`int`
           Order of the AR polynomial.

       **q** : :obj:`int`
           Order of the MA polynomial.

       **observation_indexes** : :obj:`jax.Array`
           Indexes of the observations.

       **observation_values** : :obj:`jax.Array`
           Values of the observations.

       **observation_errors** : :obj:`jax.Array`
           Errors of the observations, if None, the errors are set to sqrt(eps).

       **prediction_indexes** : :obj:`jax.Array`
           Indexes of the predictions.

       **model** : :obj:`CARMA_model`
           CARMA model.

       **kalman** : :obj:`KalmanFilter`
           Kalman filter associated to the CARMA model.

       **use_beta** : :obj:`bool`
           If True, uses the beta coefficients otherwise uses the quadratic coefficients of the MA polynomial.

       **scale_errors** : :obj:`bool`
           If True, scales the errors by a factor nu.

       **estimate_mean** : :obj:`bool`
           If True, estimates the mean of the process.

       **nb_prediction_points** : :obj:`int`
           Number of prediction points.   


   ..
       !! processed by numpydoc !!

   .. rubric:: Overview

   .. list-table:: Attributes
      :header-rows: 0
      :widths: auto
      :class: summarytable

      * - :py:obj:`p <pioran.carma.CARMAProcess.p>`
        - \-
      * - :py:obj:`q <pioran.carma.CARMAProcess.q>`
        - \-
      * - :py:obj:`observation_indexes <pioran.carma.CARMAProcess.observation_indexes>`
        - \-
      * - :py:obj:`observation_values <pioran.carma.CARMAProcess.observation_values>`
        - \-
      * - :py:obj:`observation_errors <pioran.carma.CARMAProcess.observation_errors>`
        - \-
      * - :py:obj:`prediction_indexes <pioran.carma.CARMAProcess.prediction_indexes>`
        - \-
      * - :py:obj:`model <pioran.carma.CARMAProcess.model>`
        - \-
      * - :py:obj:`kalman <pioran.carma.CARMAProcess.kalman>`
        - \-
      * - :py:obj:`use_beta <pioran.carma.CARMAProcess.use_beta>`
        - \-
      * - :py:obj:`estimate_mean <pioran.carma.CARMAProcess.estimate_mean>`
        - \-
      * - :py:obj:`scale_errors <pioran.carma.CARMAProcess.scale_errors>`
        - \-
      * - :py:obj:`nb_prediction_points <pioran.carma.CARMAProcess.nb_prediction_points>`
        - \-


   .. list-table:: Methods
      :header-rows: 0
      :widths: auto
      :class: summarytable

      * - :py:obj:`compute_predictive_distribution <pioran.carma.CARMAProcess.compute_predictive_distribution>`\ (\*\*kwargs)
        - \-
      * - :py:obj:`compute_log_marginal_likelihood <pioran.carma.CARMAProcess.compute_log_marginal_likelihood>`\ ()
        - \-
      * - :py:obj:`wrapper_log_marginal_likelihood <pioran.carma.CARMAProcess.wrapper_log_marginal_likelihood>`\ (params)
        - Wrapper to compute the log marginal likelihood in function of the (hyper)parameters.
      * - :py:obj:`__str__ <pioran.carma.CARMAProcess.__str__>`\ ()
        - String representation of the CARMA object.
      * - :py:obj:`__repr__ <pioran.carma.CARMAProcess.__repr__>`\ ()
        - Return repr(self).


   .. rubric:: Members

   .. py:attribute:: p
      :type: int

      

   .. py:attribute:: q
      :type: int

      

   .. py:attribute:: observation_indexes
      :type: jax.Array

      

   .. py:attribute:: observation_values
      :type: jax.Array

      

   .. py:attribute:: observation_errors
      :type: jax.Array

      

   .. py:attribute:: prediction_indexes
      :type: jax.Array

      

   .. py:attribute:: model
      :type: pioran.carma.carma_model.CARMA_model

      

   .. py:attribute:: kalman
      :type: pioran.carma.kalman.KalmanFilter

      

   .. py:attribute:: use_beta
      :type: bool

      

   .. py:attribute:: estimate_mean
      :type: bool

      

   .. py:attribute:: scale_errors
      :type: bool

      

   .. py:attribute:: nb_prediction_points
      :type: int

      

   .. py:method:: compute_predictive_distribution(**kwargs)


   .. py:method:: compute_log_marginal_likelihood() -> float


   .. py:method:: wrapper_log_marginal_likelihood(params) -> float

      
      Wrapper to compute the log marginal likelihood in function of the (hyper)parameters. 


      :Parameters:

          **parameters: array of shape (n)**
              (Hyper)parameters of the process.

      :Returns:

          float
              Log marginal likelihood of the CARMA process.













      ..
          !! processed by numpydoc !!

   .. py:method:: __str__() -> str

      
      String representation of the CARMA object.



      :Returns:

          str
              String representation of the CARMA object.        













      ..
          !! processed by numpydoc !!

   .. py:method:: __repr__() -> str

      
      Return repr(self).
















      ..
          !! processed by numpydoc !!



.. py:class:: CARMA_model(p, q, AR_quad=None, MA_quad=None, beta=None, use_beta=True, lorentzian_centroids=None, lorentzian_widths=None, weights=None, **kwargs)

   Bases: :py:obj:`equinox.Module`

   
   Base class for Continuous-time AutoRegressive Moving Average (CARMA) models. Inherits from eqxinox.Module.

   This class implements the basic functionality for CARMA models

   :Parameters:

       **parameters** : :obj:`ParametersModel`
           Parameters of the model.

       **p** : :obj:`int`
           Order of the AR part of the model.

       **q** : :obj:`int`
           Order of the MA part of the model. 0 <= q < p












   :Attributes:

       **parameters** : :obj:`ParametersModel`
           Parameters of the model.

       **p** : :obj:`int`
           Order of the AR part of the model.

       **q** : :obj:`int`
           Order of the MA part of the model. 0 <= q < p

       **_p** : :obj:`int`
           Order of the AR part of the model. p+1

       **_q** : :obj:`int`
           Order of the MA part of the model. q+1   


   ..
       !! processed by numpydoc !!

   .. rubric:: Overview

   .. list-table:: Attributes
      :header-rows: 0
      :widths: auto
      :class: summarytable

      * - :py:obj:`parameters <pioran.carma.CARMA_model.parameters>`
        - \-
      * - :py:obj:`ndims <pioran.carma.CARMA_model.ndims>`
        - \-
      * - :py:obj:`p <pioran.carma.CARMA_model.p>`
        - \-
      * - :py:obj:`q <pioran.carma.CARMA_model.q>`
        - \-
      * - :py:obj:`use_beta <pioran.carma.CARMA_model.use_beta>`
        - \-


   .. list-table:: Methods
      :header-rows: 0
      :widths: auto
      :class: summarytable

      * - :py:obj:`__str__ <pioran.carma.CARMA_model.__str__>`\ ()
        - String representation of the model.
      * - :py:obj:`__repr__ <pioran.carma.CARMA_model.__repr__>`\ ()
        - Return repr(self).
      * - :py:obj:`PowerSpectrum <pioran.carma.CARMA_model.PowerSpectrum>`\ (f)
        - Computes the power spectrum of the CARMA process.
      * - :py:obj:`get_AR_quads <pioran.carma.CARMA_model.get_AR_quads>`\ ()
        - Returns the quadratic coefficients of the AR part of the model.
      * - :py:obj:`get_MA_quads <pioran.carma.CARMA_model.get_MA_quads>`\ ()
        - Returns the quadratic coefficients of the MA part of the model.
      * - :py:obj:`get_AR_coeffs <pioran.carma.CARMA_model.get_AR_coeffs>`\ ()
        - Returns the coefficients of the AR part of the model.
      * - :py:obj:`get_MA_coeffs <pioran.carma.CARMA_model.get_MA_coeffs>`\ ()
        - Returns the quadratic coefficients of the AR part of the model.
      * - :py:obj:`get_AR_roots <pioran.carma.CARMA_model.get_AR_roots>`\ ()
        - Returns the roots of the AR part of the model.
      * - :py:obj:`Autocovariance <pioran.carma.CARMA_model.Autocovariance>`\ (tau)
        - Compute the autocovariance function of a CARMA(p,q) process.
      * - :py:obj:`init_statespace <pioran.carma.CARMA_model.init_statespace>`\ (y_0, errsize)
        - Initialises the state space representation of the model
      * - :py:obj:`statespace_representation <pioran.carma.CARMA_model.statespace_representation>`\ (dt)
        - \-


   .. rubric:: Members

   .. py:attribute:: parameters
      :type: pioran.parameters.ParametersModel

      

   .. py:attribute:: ndims
      :type: int

      

   .. py:attribute:: p
      :type: int

      

   .. py:attribute:: q
      :type: int

      

   .. py:attribute:: use_beta
      :type: bool

      

   .. py:method:: __str__() -> str

      
      String representation of the model.

      Also prints the roots and coefficients of the AR and MA parts of the model.















      ..
          !! processed by numpydoc !!

   .. py:method:: __repr__() -> str

      
      Return repr(self).
















      ..
          !! processed by numpydoc !!

   .. py:method:: PowerSpectrum(f: jax.Array) -> jax.Array

      
      Computes the power spectrum of the CARMA process.


      :Parameters:

          **f** : :obj:`jax.Array`
              Frequencies at which the power spectrum is evaluated.

      :Returns:

          **P** : :obj:`jax.Array`
              Power spectrum of the CARMA process.













      ..
          !! processed by numpydoc !!

   .. py:method:: get_AR_quads() -> jax.Array

      
      Returns the quadratic coefficients of the AR part of the model.

      Iterates over the parameters of the model and returns the quadratic
      coefficients of the AR part of the model.


      :Returns:

          :obj:`jax.Array`
              Quadratic coefficients of the AR part of the model.













      ..
          !! processed by numpydoc !!

   .. py:method:: get_MA_quads() -> jax.Array

      
      Returns the quadratic coefficients of the MA part of the model.

      Iterates over the parameters of the model and returns the quadratic
      coefficients of the MA part of the model.


      :Returns:

          :obj:`jax.Array`
              Quadratic coefficients of the MA part of the model.













      ..
          !! processed by numpydoc !!

   .. py:method:: get_AR_coeffs() -> jax.Array

      
      Returns the coefficients of the AR part of the model.



      :Returns:

          :obj:`jax.Array`
              Coefficients of the AR part of the model.













      ..
          !! processed by numpydoc !!

   .. py:method:: get_MA_coeffs() -> jax.Array

      
      Returns the quadratic coefficients of the AR part of the model.

      Iterates over the parameters of the model and returns the quadratic
      coefficients of the AR part of the model.


      :Returns:

          :obj:`jax.Array`
              Quadratic coefficients of the AR part of the model.













      ..
          !! processed by numpydoc !!

   .. py:method:: get_AR_roots() -> jax.Array

      
      Returns the roots of the AR part of the model.



      :Returns:

          :obj:`jax.Array`
              Roots of the AR part of the model.













      ..
          !! processed by numpydoc !!

   .. py:method:: Autocovariance(tau: jax.Array) -> jax.Array

      
      Compute the autocovariance function of a CARMA(p,q) process.
















      ..
          !! processed by numpydoc !!

   .. py:method:: init_statespace(y_0=None, errsize=None) -> Tuple[jax.Array, jax.Array] | Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]

      
      Initialises the state space representation of the model

      Parameters















      ..
          !! processed by numpydoc !!

   .. py:method:: statespace_representation(dt: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array] | jax.Array




.. py:class:: CARMA_covariance(p, q, AR_quad=None, MA_quad=None, beta=None, use_beta=True, lorentzian_centroids=None, lorentzian_widths=None, weights=None, **kwargs)

   Bases: :py:obj:`pioran.acvf.CovarianceFunction`

   
   Covariance function of a Continuous AutoRegressive Moving Average (CARMA) process.


   :Parameters:

       **p** : :obj:`int`
           Order of the AR part of the model.

       **q** : :obj:`int`
           Order of the MA part of the model. 0 <= q < p

       **AR_quad** : :obj:`list` of :obj:`float`
           Quadratic coefficients of the AR part of the model.

       **MA_quad** : :obj:`list` of :obj:`float`
           Quadratic coefficients of the MA part of the model.

       **beta** : :obj:`list` of :obj:`float`
           MA coefficients of the model.

       **use_beta** : :obj:`bool`
           If True, the MA coefficients are given by the beta parameters. If False, the MA coefficients are given by the quadratic coefficients.

       **lorentzian_centroids** : :obj:`list` of :obj:`float`
           Centroids of the Lorentzian functions.

       **lorentzian_widths** : :obj:`list` of :obj:`float`
           Widths of the Lorentzian functions.

       **weights** : :obj:`list` of :obj:`float`
           Weights of the Lorentzian functions.














   ..
       !! processed by numpydoc !!

   .. rubric:: Overview

   .. list-table:: Attributes
      :header-rows: 0
      :widths: auto
      :class: summarytable

      * - :py:obj:`parameters <pioran.carma.CARMA_covariance.parameters>`
        - Parameters of the covariance function.
      * - :py:obj:`expression <pioran.carma.CARMA_covariance.expression>`
        - Expression of the covariance function.
      * - :py:obj:`p <pioran.carma.CARMA_covariance.p>`
        - Order of the AR part of the model.
      * - :py:obj:`q <pioran.carma.CARMA_covariance.q>`
        - Order of the MA part of the model. 0 <= q < p
      * - :py:obj:`use_beta <pioran.carma.CARMA_covariance.use_beta>`
        - If True, the MA coefficients are given by the beta parameters. If False, the MA coefficients are given by the quadratic coefficients.


   .. list-table:: Methods
      :header-rows: 0
      :widths: auto
      :class: summarytable

      * - :py:obj:`get_AR_quads <pioran.carma.CARMA_covariance.get_AR_quads>`\ ()
        - Returns the quadratic coefficients of the AR part of the model.
      * - :py:obj:`get_MA_quads <pioran.carma.CARMA_covariance.get_MA_quads>`\ ()
        - Returns the quadratic coefficients of the MA part of the model.
      * - :py:obj:`get_MA_coeffs <pioran.carma.CARMA_covariance.get_MA_coeffs>`\ ()
        - Returns the quadratic coefficients of the AR part of the model.
      * - :py:obj:`get_AR_roots <pioran.carma.CARMA_covariance.get_AR_roots>`\ ()
        - Returns the roots of the AR part of the model.
      * - :py:obj:`calculate <pioran.carma.CARMA_covariance.calculate>`\ (tau)
        - Compute the autocovariance function of a CARMA(p,q) process.


   .. rubric:: Members

   .. py:attribute:: parameters
      :type: pioran.parameters.ParametersModel

      
      Parameters of the covariance function.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: expression
      :type: str

      
      Expression of the covariance function.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: p
      :type: int

      
      Order of the AR part of the model.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: q
      :type: int

      
      Order of the MA part of the model. 0 <= q < p
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: use_beta
      :type: bool

      
      If True, the MA coefficients are given by the beta parameters. If False, the MA coefficients are given by the quadratic coefficients.
















      ..
          !! processed by numpydoc !!

   .. py:method:: get_AR_quads() -> jax.Array

      
      Returns the quadratic coefficients of the AR part of the model.

      Iterates over the parameters of the model and returns the quadratic
      coefficients of the AR part of the model.


      :Returns:

          :obj:`jax.Array`
              Quadratic coefficients of the AR part of the model.













      ..
          !! processed by numpydoc !!

   .. py:method:: get_MA_quads() -> jax.Array

      
      Returns the quadratic coefficients of the MA part of the model.

      Iterates over the parameters of the model and returns the quadratic
      coefficients of the MA part of the model.


      :Returns:

          :obj:`jax.Array`
              Quadratic coefficients of the MA part of the model.













      ..
          !! processed by numpydoc !!

   .. py:method:: get_MA_coeffs() -> jax.Array

      
      Returns the quadratic coefficients of the AR part of the model.

      Iterates over the parameters of the model and returns the quadratic
      coefficients of the AR part of the model.


      :Returns:

          :obj:`jax.Array`
              Quadratic coefficients of the AR part of the model.













      ..
          !! processed by numpydoc !!

   .. py:method:: get_AR_roots() -> jax.Array

      
      Returns the roots of the AR part of the model.



      :Returns:

          :obj:`jax.Array`
              Roots of the AR part of the model.













      ..
          !! processed by numpydoc !!

   .. py:method:: calculate(tau: jax.Array) -> jax.Array

      
      Compute the autocovariance function of a CARMA(p,q) process.
















      ..
          !! processed by numpydoc !!



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

      * - :py:obj:`observation_indexes <pioran.carma.KalmanFilter.observation_indexes>`
        - \-
      * - :py:obj:`observation_values <pioran.carma.KalmanFilter.observation_values>`
        - \-
      * - :py:obj:`observation_errors <pioran.carma.KalmanFilter.observation_errors>`
        - \-
      * - :py:obj:`model <pioran.carma.KalmanFilter.model>`
        - \-


   .. list-table:: Methods
      :header-rows: 0
      :widths: auto
      :class: summarytable

      * - :py:obj:`Predict <pioran.carma.KalmanFilter.Predict>`\ (X, P, F, Q)
        - Predict step of the Kalman filter.
      * - :py:obj:`Update <pioran.carma.KalmanFilter.Update>`\ (X, P, Z, H, R)
        - Update step of the Kalman filter.
      * - :py:obj:`one_step_loglike_CAR1 <pioran.carma.KalmanFilter.one_step_loglike_CAR1>`\ (carry, xs)
        - Compute the log-likelihood of a single observation value.
      * - :py:obj:`one_step_loglike_CARMA <pioran.carma.KalmanFilter.one_step_loglike_CARMA>`\ (carry, xs)
        - Compute the log-likelihood of a single observation value.
      * - :py:obj:`log_likelihood <pioran.carma.KalmanFilter.log_likelihood>`\ ()
        - \-
      * - :py:obj:`wrapper_log_marginal_likelihood <pioran.carma.KalmanFilter.wrapper_log_marginal_likelihood>`\ (params)
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







