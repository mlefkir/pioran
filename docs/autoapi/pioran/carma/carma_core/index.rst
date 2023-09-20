
carma_core
==========

.. py:module:: pioran.carma.carma_core


Overview
--------

.. list-table:: Classes
   :header-rows: 0
   :widths: auto
   :class: summarytable

   * - :py:obj:`CARMAProcess <pioran.carma.carma_core.CARMAProcess>`
     - Base class for inference with Continuous autoregressive moving average processes




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

      * - :py:obj:`p <pioran.carma.carma_core.CARMAProcess.p>`
        - \-
      * - :py:obj:`q <pioran.carma.carma_core.CARMAProcess.q>`
        - \-
      * - :py:obj:`observation_indexes <pioran.carma.carma_core.CARMAProcess.observation_indexes>`
        - \-
      * - :py:obj:`observation_values <pioran.carma.carma_core.CARMAProcess.observation_values>`
        - \-
      * - :py:obj:`observation_errors <pioran.carma.carma_core.CARMAProcess.observation_errors>`
        - \-
      * - :py:obj:`prediction_indexes <pioran.carma.carma_core.CARMAProcess.prediction_indexes>`
        - \-
      * - :py:obj:`model <pioran.carma.carma_core.CARMAProcess.model>`
        - \-
      * - :py:obj:`kalman <pioran.carma.carma_core.CARMAProcess.kalman>`
        - \-
      * - :py:obj:`use_beta <pioran.carma.carma_core.CARMAProcess.use_beta>`
        - \-
      * - :py:obj:`estimate_mean <pioran.carma.carma_core.CARMAProcess.estimate_mean>`
        - \-
      * - :py:obj:`scale_errors <pioran.carma.carma_core.CARMAProcess.scale_errors>`
        - \-
      * - :py:obj:`nb_prediction_points <pioran.carma.carma_core.CARMAProcess.nb_prediction_points>`
        - \-


   .. list-table:: Methods
      :header-rows: 0
      :widths: auto
      :class: summarytable

      * - :py:obj:`compute_predictive_distribution <pioran.carma.carma_core.CARMAProcess.compute_predictive_distribution>`\ (\*\*kwargs)
        - \-
      * - :py:obj:`compute_log_marginal_likelihood <pioran.carma.carma_core.CARMAProcess.compute_log_marginal_likelihood>`\ ()
        - \-
      * - :py:obj:`wrapper_log_marginal_likelihood <pioran.carma.carma_core.CARMAProcess.wrapper_log_marginal_likelihood>`\ (params)
        - Wrapper to compute the log marginal likelihood in function of the (hyper)parameters.
      * - :py:obj:`__str__ <pioran.carma.carma_core.CARMAProcess.__str__>`\ ()
        - String representation of the CARMA object.
      * - :py:obj:`__repr__ <pioran.carma.carma_core.CARMAProcess.__repr__>`\ ()
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






