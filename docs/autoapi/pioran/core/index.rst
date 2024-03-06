
core
====

.. py:module:: pioran.core

.. autoapi-nested-parse::

   Gaussian process regression for time series analysis.

   ..
       !! processed by numpydoc !!


Overview
--------

.. list-table:: Classes
   :header-rows: 0
   :widths: auto
   :class: summarytable

   * - :py:obj:`GaussianProcess <pioran.core.GaussianProcess>`
     - Gaussian Process Regression of univariate time series.



.. list-table:: Attributes
   :header-rows: 0
   :widths: auto
   :class: summarytable

   * - :py:obj:`celerite <pioran.core.celerite>`
     - \-


Classes
-------

.. py:class:: GaussianProcess(function: pioran.acvf_base.CovarianceFunction | pioran.psd_base.PowerSpectralDensity, observation_indexes: jax.Array, observation_values: jax.Array, observation_errors: jax.Array | None = None, S_low: float = 10, S_high: float = 10, method: str = 'FFT', use_tinygp: bool = False, n_components: int = 0, estimate_variance: bool = True, estimate_mean: bool = True, scale_errors: bool = True, log_transform: bool = False, nb_prediction_points: int = 0, propagate_errors: bool = True, prediction_indexes: jax.Array | None = None, use_celerite: bool = False, use_legacy_celerite: bool = False)

   Bases: :py:obj:`equinox.Module`

   
   Gaussian Process Regression of univariate time series.

   Model the time series as a Gaussian Process with a given covariance function or power spectral density. The covariance function
   is given by a :class:`~pioran.acvf_base.CovarianceFunction` object and the power spectral density is given by a :class:`~pioran.psd_base.PowerSpectralDensity` object.

   :Parameters:

       **function** : :class:`~pioran.acvf_base.CovarianceFunction` or :class:`~pioran.psd_base.PowerSpectralDensity`
           Model function associated to the Gaussian Process. Can be a covariance function or a power spectral density.

       **observation_indexes** : :obj:`jax.Array`
           Indexes of the observed data, in this case it is the time.

       **observation_values** : :obj:`jax.Array`
           Observation data.

       **observation_errors** : :obj:`jax.Array`, optional
           Errors on the observables, by default :obj:`None`

       **S_low** : :obj:`float`, optional
           Scaling factor to extend the frequency grid to lower values, by default 10. See :obj:`PSDToACV` for more details.

       **S_high** : :obj:`float`, optional
           Scaling factor to extend the frequency grid to higher values, by default 10. See :obj:`PSDToACV` for more details.

       **method** : :obj:`str`, optional
           Method to compute the covariance function from the power spectral density, by default 'FFT'.
           Possible values are:
           - 'FFT': use the FFT to compute the autocovariance function.
           - 'NuFFT': use the non-uniform FFT to compute the autocovariance function.
           - 'SHO': approximate the power spectrum as a sum of SHO basis functions to compute the autocovariance function.
           - 'DRWCelerite': approximate the power spectrum as a sum of DRW+Celerite basis functions to compute the autocovariance function.

       **use_tinygp** : :obj:`bool`, optional
           Use tinygp to compute the log marginal likelihood, by default False. Should only be used when the power spectrum model
           is expressed as a sum of quasi-separable kernels, i.e. method is not 'FFT' or 'NuFFT'.

       **n_components** : :obj:`int`, optional
           Number of components to use when using tinygp and the power spectrum model is expressed as a sum of quasi-separable kernels.

       **estimate_mean** : :obj:`bool`, optional
           Estimate the mean of the observed data, by default True.

       **estimate_variance** : :obj:`bool`, optional
           Estimate the amplitude of the autocovariance function, by default True.

       **scale_errors** : :obj:`bool`, optional
           Scale the errors on the observed data by adding a multiplicative factor, by default True.

       **log_transform** : :obj:`bool`, optional
           Use a log transformation of the data, by default False. This is useful when the data is log-normal distributed.
           Only compatible with the method 'FFT' or 'NuFFT'.

       **nb_prediction_points** : :obj:`int`, optional
           Number of points to predict, by default 5 * length(observed(indexes)).

       **prediction_indexes** : :obj:`jax.Array`, optional
           indexes of the prediction data, by default linspace(min(observation_indexes),max(observation_indexes),nb_prediction_points)














   ..
       !! processed by numpydoc !!

   .. rubric:: Overview

   .. list-table:: Attributes
      :header-rows: 0
      :widths: auto
      :class: summarytable

      * - :py:obj:`model <pioran.core.GaussianProcess.model>`
        - Model associated to the Gaussian Process, can be a covariance function or a power spectral density to autocovariance function converter.
      * - :py:obj:`observation_indexes <pioran.core.GaussianProcess.observation_indexes>`
        - Indexes of the observed data, in this case it is the time.
      * - :py:obj:`observation_errors <pioran.core.GaussianProcess.observation_errors>`
        - Errors on the observed data.
      * - :py:obj:`observation_values <pioran.core.GaussianProcess.observation_values>`
        - Observed data.
      * - :py:obj:`prediction_indexes <pioran.core.GaussianProcess.prediction_indexes>`
        - Indexes of the prediction data.
      * - :py:obj:`nb_prediction_points <pioran.core.GaussianProcess.nb_prediction_points>`
        - Number of points to predict, by default 5 * length(observed(indexes)).
      * - :py:obj:`scale_errors <pioran.core.GaussianProcess.scale_errors>`
        - Scale the errors on the observed data by adding a multiplicative factor.
      * - :py:obj:`estimate_mean <pioran.core.GaussianProcess.estimate_mean>`
        - Estimate the mean of the observed data.
      * - :py:obj:`estimate_variance <pioran.core.GaussianProcess.estimate_variance>`
        - Estimate the amplitude of the autocovariance function.
      * - :py:obj:`log_transform <pioran.core.GaussianProcess.log_transform>`
        - Use a log transformation of the data.
      * - :py:obj:`use_tinygp <pioran.core.GaussianProcess.use_tinygp>`
        - Use tinygp to compute the log marginal likelihood.
      * - :py:obj:`propagate_errors <pioran.core.GaussianProcess.propagate_errors>`
        - Propagate the errors on the observed data.
      * - :py:obj:`use_celerite <pioran.core.GaussianProcess.use_celerite>`
        - Use celerite2-jax as a backend to model the autocovariance function and compute the log marginal likelihood.
      * - :py:obj:`use_legacy_celerite <pioran.core.GaussianProcess.use_legacy_celerite>`
        - Use celerite2 as a backend to model the autocovariance function and compute the log marginal likelihood.


   .. list-table:: Methods
      :header-rows: 0
      :widths: auto
      :class: summarytable

      * - :py:obj:`get_cov <pioran.core.GaussianProcess.get_cov>`\ (xt, xp, errors)
        - Compute the covariance matrix between two arrays.
      * - :py:obj:`get_cov_training <pioran.core.GaussianProcess.get_cov_training>`\ ()
        - Compute the covariance matrix and other vectors for the observed data.
      * - :py:obj:`compute_predictive_distribution <pioran.core.GaussianProcess.compute_predictive_distribution>`\ (log_transform, prediction_indexes)
        - Compute the predictive mean and the predictive covariance of the GP.
      * - :py:obj:`compute_log_marginal_likelihood_pioran <pioran.core.GaussianProcess.compute_log_marginal_likelihood_pioran>`\ ()
        - Compute the log marginal likelihood of the Gaussian Process.
      * - :py:obj:`build_gp_celerite <pioran.core.GaussianProcess.build_gp_celerite>`\ ()
        - Build the Gaussian Process using :obj:`celerite2.jax`.
      * - :py:obj:`build_gp_celerite_legacy <pioran.core.GaussianProcess.build_gp_celerite_legacy>`\ ()
        - Build the Gaussian Process using :obj:`celerite2`.
      * - :py:obj:`build_gp_tinygp <pioran.core.GaussianProcess.build_gp_tinygp>`\ ()
        - Build the Gaussian Process using :obj:`tinygp`.
      * - :py:obj:`compute_log_marginal_likelihood_celerite <pioran.core.GaussianProcess.compute_log_marginal_likelihood_celerite>`\ ()
        - Compute the log marginal likelihood of the Gaussian Process using celerite.
      * - :py:obj:`compute_log_marginal_likelihood_tinygp <pioran.core.GaussianProcess.compute_log_marginal_likelihood_tinygp>`\ ()
        - Compute the log marginal likelihood of the Gaussian Process using tinygp.
      * - :py:obj:`compute_log_marginal_likelihood <pioran.core.GaussianProcess.compute_log_marginal_likelihood>`\ ()
        - \-
      * - :py:obj:`wrapper_log_marginal_likelihood <pioran.core.GaussianProcess.wrapper_log_marginal_likelihood>`\ (parameters)
        - Wrapper to compute the log marginal likelihood in function of the (hyper)parameters.
      * - :py:obj:`wrapper_neg_log_marginal_likelihood <pioran.core.GaussianProcess.wrapper_neg_log_marginal_likelihood>`\ (parameters)
        - Wrapper to compute the negative log marginal likelihood in function of the (hyper)parameters.
      * - :py:obj:`__str__ <pioran.core.GaussianProcess.__str__>`\ ()
        - String representation of the GP object.
      * - :py:obj:`__repr__ <pioran.core.GaussianProcess.__repr__>`\ ()
        - Return repr(self).


   .. rubric:: Members

   .. py:attribute:: model
      :type: pioran.acvf_base.CovarianceFunction | pioran.psdtoacv.PSDToACV

      
      Model associated to the Gaussian Process, can be a covariance function or a power spectral density to autocovariance function converter.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: observation_indexes
      :type: jax.Array

      
      Indexes of the observed data, in this case it is the time.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: observation_errors
      :type: jax.Array

      
      Errors on the observed data.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: observation_values
      :type: jax.Array

      
      Observed data.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: prediction_indexes
      :type: jax.Array

      
      Indexes of the prediction data.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: nb_prediction_points
      :type: int

      
      Number of points to predict, by default 5 * length(observed(indexes)).
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: scale_errors
      :type: bool
      :value: True

      
      Scale the errors on the observed data by adding a multiplicative factor.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: estimate_mean
      :type: bool
      :value: True

      
      Estimate the mean of the observed data.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: estimate_variance
      :type: bool
      :value: False

      
      Estimate the amplitude of the autocovariance function.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: log_transform
      :type: bool
      :value: False

      
      Use a log transformation of the data.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: use_tinygp
      :type: bool
      :value: False

      
      Use tinygp to compute the log marginal likelihood.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: propagate_errors
      :type: bool
      :value: True

      
      Propagate the errors on the observed data.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: use_celerite
      :type: bool
      :value: False

      
      Use celerite2-jax as a backend to model the autocovariance function and compute the log marginal likelihood.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: use_legacy_celerite
      :type: bool
      :value: False

      
      Use celerite2 as a backend to model the autocovariance function and compute the log marginal likelihood.
















      ..
          !! processed by numpydoc !!

   .. py:method:: get_cov(xt: jax.Array, xp: jax.Array, errors: jax.Array | None = None) -> jax.Array

      
      Compute the covariance matrix between two arrays.

      To compute the covariance matrix, this function calls the get_cov_matrix method of the model.
      If the errors are not None, then the covariance matrix is computed for the observationst,
      i.e. with observed data as input (xt=xp=observed data) and the errors on the measurement.
      The total covariance matrix is computed as:

      .. math::

          C = K + \nu \sigma ^ 2 \times [I]

      With :math:`I` the identity matrix, :math:`K` the covariance matrix, :math:`\sigma` the errors and :math:`\nu` a free parameter to scale the errors.

      :Parameters:

          **xt: :obj:`jax.Array`**
              First array.

          **xp: :obj:`jax.Array`**
              Second array.

          **errors: :obj:`jax.Array`, optional**
              Errors on the observed data

      :Returns:

          :obj:`jax.Array`
              Covariance matrix between the two arrays.













      ..
          !! processed by numpydoc !!

   .. py:method:: get_cov_training() -> tuple[jax.Array, jax.Array, jax.Array]

      
      Compute the covariance matrix and other vectors for the observed data.



      :Returns:

          :obj:`jax.Array`
              Covariance matrix for the observed data.

          :obj:`jax.Array`
              Inverse of Cov_xx.

          :obj:`jax.Array`
              alpha = Cov_inv * observation_values (- mu if mu is estimated)













      ..
          !! processed by numpydoc !!

   .. py:method:: compute_predictive_distribution(log_transform: bool | None = None, prediction_indexes: jax.Array | None = None)

      
      Compute the predictive mean and the predictive covariance of the GP.

      The predictive distribution are computed using equations (2.25)  and (2.26) in Rasmussen and Williams (2006).

      :Parameters:

          **log_transform: bool or None, optional**
              Predict using a with exponentation of the posterior mean, by default use the default value of the GP.

          **prediction_indexes: array of length m, optional**
              Indexes of the prediction data, by default jnp.linspace(jnp.min(observation_indexes),jnp.max(observation_indexes),nb_prediction_points)

      :Returns:

          :obj:`jax.Array`
              Predictive mean of the GP.

          :obj:`jax.Array`
              Predictive covariance of the GP.













      ..
          !! processed by numpydoc !!

   .. py:method:: compute_log_marginal_likelihood_pioran() -> float

      
      Compute the log marginal likelihood of the Gaussian Process.

      The log marginal likelihood is computed using algorithm (2.1) in Rasmussen and Williams (2006)
      Following the notation of the book, :math:`x` are the observed indexes, x* is the predictive indexes, y the observations,
      k the covariance function, sigma the errors on the observations.

      Solve of triangular system instead of inverting the matrix:

      :math:`L = {\rm cholesky}( k(x,x) + \nu \sigma^2 \times [I] )`

      :math:`z = L^{-1} \times (\boldsymbol{y}-\mu))`

      :math:`\mathcal{L} = - \frac{1}{2} z^T z - \sum_i \log L_{ii} - \frac{n}{2} \log (2 \pi)`


      :Returns:

          :obj:`float`
              Log marginal likelihood of the GP.













      ..
          !! processed by numpydoc !!

   .. py:method:: build_gp_celerite()

      
      Build the Gaussian Process using :obj:`celerite2.jax`.

      This function is called when the power spectrum model is expressed as a sum of quasi-separable kernels.
      In this case, the covariance function is a sum of :obj:`celerite2.jax.terms` objects.


      :Returns:

          :obj:`tinygp.GaussianProcess`
              Gaussian Process object.













      ..
          !! processed by numpydoc !!

   .. py:method:: build_gp_celerite_legacy()

      
      Build the Gaussian Process using :obj:`celerite2`.

      This function is called when the power spectrum model is expressed as a sum of quasi-separable kernels.
      In this case, the covariance function is a sum of :obj:`tinygp.kernels.quasisep` objects.


      :Returns:

          :obj:`tinygp.GaussianProcess`
              Gaussian Process object.













      ..
          !! processed by numpydoc !!

   .. py:method:: build_gp_tinygp() -> tinygp.GaussianProcess

      
      Build the Gaussian Process using :obj:`tinygp`.

      This function is called when the power spectrum model is expressed as a sum of quasi-separable kernels.
      In this case, the covariance function is a sum of :obj:`tinygp.kernels.quasisep` objects.


      :Returns:

          :obj:`tinygp.GaussianProcess`
              Gaussian Process object.













      ..
          !! processed by numpydoc !!

   .. py:method:: compute_log_marginal_likelihood_celerite() -> jax.Array

      
      Compute the log marginal likelihood of the Gaussian Process using celerite.

      This function is called when the power spectrum model is expressed as a sum of quasi-separable kernels.
      In this case, the covariance function is a sum of :obj:`celerite2.jax.Terms` objects.


      :Returns:

          :obj:`float`
              Log marginal likelihood of the GP.













      ..
          !! processed by numpydoc !!

   .. py:method:: compute_log_marginal_likelihood_tinygp() -> jax.Array

      
      Compute the log marginal likelihood of the Gaussian Process using tinygp.

      This function is called when the power spectrum model is expressed as a sum of quasi-separable kernels.
      In this case, the covariance function is a sum of :obj:`tinygp.kernels.quasisep` objects.


      :Returns:

          :obj:`float`
              Log marginal likelihood of the GP.













      ..
          !! processed by numpydoc !!

   .. py:method:: compute_log_marginal_likelihood() -> float


   .. py:method:: wrapper_log_marginal_likelihood(parameters: jax.Array) -> float

      
      Wrapper to compute the log marginal likelihood in function of the (hyper)parameters.


      :Parameters:

          **parameters: :obj:`jax.Array`**
              (Hyper)parameters of the covariance function.

      :Returns:

          :obj:`float`
              Log marginal likelihood of the GP.













      ..
          !! processed by numpydoc !!

   .. py:method:: wrapper_neg_log_marginal_likelihood(parameters: jax.Array) -> float

      
      Wrapper to compute the negative log marginal likelihood in function of the (hyper)parameters.


      :Parameters:

          **parameters: :obj:`jax.Array` of shape (n)**
              (Hyper)parameters of the covariance function.

      :Returns:

          float
              Negative log marginal likelihood of the GP.













      ..
          !! processed by numpydoc !!

   .. py:method:: __str__() -> str

      
      String representation of the GP object.



      :Returns:

          :obj:`str`
              String representation of the GP object.













      ..
          !! processed by numpydoc !!

   .. py:method:: __repr__() -> str

      
      Return repr(self).
















      ..
          !! processed by numpydoc !!




Attributes
----------
.. py:data:: celerite

   



