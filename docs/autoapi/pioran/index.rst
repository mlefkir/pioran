
pioran
======

.. py:module:: pioran


Subpackages
-----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   carma/index.rst
   parameters/index.rst
   utils/index.rst


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   acvf/index.rst
   acvf_base/index.rst
   core/index.rst
   diagnostics/index.rst
   inference/index.rst
   plots/index.rst
   priors/index.rst
   psd/index.rst
   psd_base/index.rst
   psdtoacv/index.rst
   simulate/index.rst
   tools/index.rst


Overview
--------

.. list-table:: Classes
   :header-rows: 0
   :widths: auto
   :class: summarytable

   * - :py:obj:`GaussianProcess <pioran.GaussianProcess>`
     - Gaussian Process Regression of univariate time series.
   * - :py:obj:`CovarianceFunction <pioran.CovarianceFunction>`
     - Represents a covariance function model.
   * - :py:obj:`PowerSpectralDensity <pioran.PowerSpectralDensity>`
     - Represents a power density function function.
   * - :py:obj:`PSDToACV <pioran.PSDToACV>`
     - Represents the tranformation of a power spectral density to an autocovariance function.
   * - :py:obj:`Inference <pioran.Inference>`
     - Class to infer the value of the (hyper)parameters of the Gaussian Process.
   * - :py:obj:`Simulations <pioran.Simulations>`
     - Simulate time series from a given PSD or ACVF model.
   * - :py:obj:`Visualisations <pioran.Visualisations>`
     - Class for visualising the results after an inference run.



.. list-table:: Attributes
   :header-rows: 0
   :widths: auto
   :class: summarytable

   * - :py:obj:`__author__ <pioran.__author__>`
     - \-
   * - :py:obj:`__version__ <pioran.__version__>`
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

      * - :py:obj:`model <pioran.GaussianProcess.model>`
        - Model associated to the Gaussian Process, can be a covariance function or a power spectral density to autocovariance function converter.
      * - :py:obj:`observation_indexes <pioran.GaussianProcess.observation_indexes>`
        - Indexes of the observed data, in this case it is the time.
      * - :py:obj:`observation_errors <pioran.GaussianProcess.observation_errors>`
        - Errors on the observed data.
      * - :py:obj:`observation_values <pioran.GaussianProcess.observation_values>`
        - Observed data.
      * - :py:obj:`prediction_indexes <pioran.GaussianProcess.prediction_indexes>`
        - Indexes of the prediction data.
      * - :py:obj:`nb_prediction_points <pioran.GaussianProcess.nb_prediction_points>`
        - Number of points to predict, by default 5 * length(observed(indexes)).
      * - :py:obj:`scale_errors <pioran.GaussianProcess.scale_errors>`
        - Scale the errors on the observed data by adding a multiplicative factor.
      * - :py:obj:`estimate_mean <pioran.GaussianProcess.estimate_mean>`
        - Estimate the mean of the observed data.
      * - :py:obj:`estimate_variance <pioran.GaussianProcess.estimate_variance>`
        - Estimate the amplitude of the autocovariance function.
      * - :py:obj:`log_transform <pioran.GaussianProcess.log_transform>`
        - Use a log transformation of the data.
      * - :py:obj:`use_tinygp <pioran.GaussianProcess.use_tinygp>`
        - Use tinygp to compute the log marginal likelihood.
      * - :py:obj:`propagate_errors <pioran.GaussianProcess.propagate_errors>`
        - Propagate the errors on the observed data.
      * - :py:obj:`use_celerite <pioran.GaussianProcess.use_celerite>`
        - Use celerite2-jax as a backend to model the autocovariance function and compute the log marginal likelihood.
      * - :py:obj:`use_legacy_celerite <pioran.GaussianProcess.use_legacy_celerite>`
        - Use celerite2 as a backend to model the autocovariance function and compute the log marginal likelihood.


   .. list-table:: Methods
      :header-rows: 0
      :widths: auto
      :class: summarytable

      * - :py:obj:`get_cov <pioran.GaussianProcess.get_cov>`\ (xt, xp, errors)
        - Compute the covariance matrix between two arrays.
      * - :py:obj:`get_cov_training <pioran.GaussianProcess.get_cov_training>`\ ()
        - Compute the covariance matrix and other vectors for the observed data.
      * - :py:obj:`compute_predictive_distribution <pioran.GaussianProcess.compute_predictive_distribution>`\ (log_transform, prediction_indexes)
        - Compute the predictive mean and the predictive covariance of the GP.
      * - :py:obj:`compute_log_marginal_likelihood_pioran <pioran.GaussianProcess.compute_log_marginal_likelihood_pioran>`\ ()
        - Compute the log marginal likelihood of the Gaussian Process.
      * - :py:obj:`build_gp_celerite <pioran.GaussianProcess.build_gp_celerite>`\ ()
        - Build the Gaussian Process using :obj:`celerite2.jax`.
      * - :py:obj:`build_gp_celerite_legacy <pioran.GaussianProcess.build_gp_celerite_legacy>`\ ()
        - Build the Gaussian Process using :obj:`celerite2`.
      * - :py:obj:`build_gp_tinygp <pioran.GaussianProcess.build_gp_tinygp>`\ ()
        - Build the Gaussian Process using :obj:`tinygp`.
      * - :py:obj:`compute_log_marginal_likelihood_celerite <pioran.GaussianProcess.compute_log_marginal_likelihood_celerite>`\ ()
        - Compute the log marginal likelihood of the Gaussian Process using celerite.
      * - :py:obj:`compute_log_marginal_likelihood_tinygp <pioran.GaussianProcess.compute_log_marginal_likelihood_tinygp>`\ ()
        - Compute the log marginal likelihood of the Gaussian Process using tinygp.
      * - :py:obj:`compute_log_marginal_likelihood <pioran.GaussianProcess.compute_log_marginal_likelihood>`\ ()
        - \-
      * - :py:obj:`wrapper_log_marginal_likelihood <pioran.GaussianProcess.wrapper_log_marginal_likelihood>`\ (parameters)
        - Wrapper to compute the log marginal likelihood in function of the (hyper)parameters.
      * - :py:obj:`wrapper_neg_log_marginal_likelihood <pioran.GaussianProcess.wrapper_neg_log_marginal_likelihood>`\ (parameters)
        - Wrapper to compute the negative log marginal likelihood in function of the (hyper)parameters.
      * - :py:obj:`__str__ <pioran.GaussianProcess.__str__>`\ ()
        - String representation of the GP object.
      * - :py:obj:`__repr__ <pioran.GaussianProcess.__repr__>`\ ()
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



.. py:class:: CovarianceFunction(param_values: pioran.parameters.ParametersModel | list[float], param_names: list[str], free_parameters: list[bool])

   Bases: :py:obj:`equinox.Module`

   
   Represents a covariance function model.

   Bridge between the parameters and the covariance function model. All covariance functions
   inherit from this class.

   :Parameters:

       **param_values** : :class:`~pioran.parameters.ParametersModel` or  :obj:`list` of :obj:`float`
           Values of the parameters of the covariance function.

       **param_names** :  :obj:`list` of :obj:`str`
           param_names of the parameters of the covariance function.

       **free_parameters** :  :obj:`list` of :obj:`bool`
           list` of :obj:`bool` to indicate if the parameters are free or not.





   :Raises:

       `TypeError`
           If param_values is not a :obj:`list` of :obj:`float` or a :class:`~pioran.parameters.ParametersModel`.









   ..
       !! processed by numpydoc !!

   .. rubric:: Overview

   .. list-table:: Attributes
      :header-rows: 0
      :widths: auto
      :class: summarytable

      * - :py:obj:`parameters <pioran.CovarianceFunction.parameters>`
        - Parameters of the covariance function.
      * - :py:obj:`expression <pioran.CovarianceFunction.expression>`
        - Expression of the covariance function.


   .. list-table:: Methods
      :header-rows: 0
      :widths: auto
      :class: summarytable

      * - :py:obj:`__str__ <pioran.CovarianceFunction.__str__>`\ ()
        - String representation of the covariance function.
      * - :py:obj:`__repr__ <pioran.CovarianceFunction.__repr__>`\ ()
        - Representation of the covariance function.
      * - :py:obj:`get_cov_matrix <pioran.CovarianceFunction.get_cov_matrix>`\ (xq, xp)
        - Compute the covariance matrix between two arrays xq, xp.
      * - :py:obj:`__add__ <pioran.CovarianceFunction.__add__>`\ (other)
        - Overload of the + operator to add two covariance functions.
      * - :py:obj:`__mul__ <pioran.CovarianceFunction.__mul__>`\ (other)
        - Overload of the * operator to multiply two covariance functions.


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

   .. py:method:: __str__() -> str

      
      String representation of the covariance function.



      :Returns:

          :obj:`str`
              String representation of the covariance function.
              Include the representation of the parameters.













      ..
          !! processed by numpydoc !!

   .. py:method:: __repr__() -> str

      
      Representation of the covariance function.



      :Returns:

          :obj:`str`
              Representation of the covariance function.
              Include the representation of the parameters.













      ..
          !! processed by numpydoc !!

   .. py:method:: get_cov_matrix(xq: jax.Array, xp: jax.Array) -> jax.Array

      
      Compute the covariance matrix between two arrays xq, xp.

      The term (xq-xp) is computed using the :func:`~pioran.utils.EuclideanDistance` function from the utils module.

      :Parameters:

          **xq** : :obj:`jax.Array`
              First array.

          **xp** : :obj:`jax.Array`
              Second array.

      :Returns:

          (N,M) :obj:`jax.Array`
              Covariance matrix.













      ..
          !! processed by numpydoc !!

   .. py:method:: __add__(other: CovarianceFunction) -> SumCovarianceFunction

      
      Overload of the + operator to add two covariance functions.


      :Parameters:

          **other** : :obj:`CovarianceFunction`
              Covariance function to add.

      :Returns:

          :obj:`SumCovarianceFunction`
              Sum of the two covariance functions.













      ..
          !! processed by numpydoc !!

   .. py:method:: __mul__(other: CovarianceFunction) -> ProductCovarianceFunction

      
      Overload of the * operator to multiply two covariance functions.


      :Parameters:

          **other** : :obj:`CovarianceFunction`
              Covariance function to multiply.

      :Returns:

          :obj:`ProductCovarianceFunction`
              Product of the two covariance functions.













      ..
          !! processed by numpydoc !!



.. py:class:: PowerSpectralDensity(param_values: pioran.parameters.ParametersModel | list[float], param_names: list[str], free_parameters: list[bool])

   Bases: :py:obj:`equinox.Module`

   
   Represents a power density function function.

   Bridge between the parameters and the power spectral density function. All power spectral density functions
   inherit from this class.

   :Parameters:

       **param_values** : :class:`~pioran.parameters.ParametersModel` or  :obj:`list` of :obj:`float`
           Values of the parameters of the power spectral density function.

       **param_names** : :obj:`list` of :obj:`str`
           param_names of the parameters of the power spectral density function.

       **free_parameters** :  :obj:`list` of :obj:`bool`
           List of bool to indicate if the parameters are free or not.





   :Raises:

       `TypeError`
           If param_values is not a :obj:`list` of `float` or a :class:`~pioran.parameters.ParametersModel`.









   ..
       !! processed by numpydoc !!

   .. rubric:: Overview

   .. list-table:: Attributes
      :header-rows: 0
      :widths: auto
      :class: summarytable

      * - :py:obj:`parameters <pioran.PowerSpectralDensity.parameters>`
        - Parameters of the power spectral density function.
      * - :py:obj:`expression <pioran.PowerSpectralDensity.expression>`
        - Expression of the power spectral density function.
      * - :py:obj:`analytical <pioran.PowerSpectralDensity.analytical>`
        - If True, the power spectral density function is analytical, otherwise it is not.


   .. list-table:: Methods
      :header-rows: 0
      :widths: auto
      :class: summarytable

      * - :py:obj:`__str__ <pioran.PowerSpectralDensity.__str__>`\ ()
        - String representation of the power spectral density.
      * - :py:obj:`__repr__ <pioran.PowerSpectralDensity.__repr__>`\ ()
        - Return repr(self).
      * - :py:obj:`__add__ <pioran.PowerSpectralDensity.__add__>`\ (other)
        - Overload of the + operator for the power spectral densities.
      * - :py:obj:`__mul__ <pioran.PowerSpectralDensity.__mul__>`\ (other)
        - Overload of the * operator for the power spectral densities.


   .. rubric:: Members

   .. py:attribute:: parameters
      :type: pioran.parameters.ParametersModel

      
      Parameters of the power spectral density function.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: expression
      :type: str

      
      Expression of the power spectral density function.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: analytical
      :type: bool
      :value: False

      
      If True, the power spectral density function is analytical, otherwise it is not.
















      ..
          !! processed by numpydoc !!

   .. py:method:: __str__() -> str

      
      String representation of the power spectral density.



      :Returns:

          :obj:`str`
              String representation of the power spectral density.













      ..
          !! processed by numpydoc !!

   .. py:method:: __repr__() -> str

      
      Return repr(self).
















      ..
          !! processed by numpydoc !!

   .. py:method:: __add__(other: PowerSpectralDensity) -> SumPowerSpectralDensity

      
      Overload of the + operator for the power spectral densities.


      :Parameters:

          **other** : :obj:`PowerSpectralDensity`
              Power spectral density to add.

      :Returns:

          :obj:`SumPowerSpectralDensity`
              Sum of the two power spectral densities.













      ..
          !! processed by numpydoc !!

   .. py:method:: __mul__(other) -> ProductPowerSpectralDensity

      
      Overload of the * operator for the power spectral densities.


      :Parameters:

          **other** : :obj:`PowerSpectralDensity`
              Power spectral density to multiply.

      :Returns:

          :obj:`ProductPowerSpectralDensity`
              Product of the two power spectral densities.













      ..
          !! processed by numpydoc !!



.. py:class:: PSDToACV(PSD: pioran.psd_base.PowerSpectralDensity, S_low: float, S_high: float, T: float, dt: float, method: str, n_components: int = 0, estimate_variance: bool = True, init_variance: float = 1.0, use_celerite=False, use_legacy_celerite: bool = False)

   Bases: :py:obj:`equinox.Module`

   
   Represents the tranformation of a power spectral density to an autocovariance function.

   Computes the autocovariance function from a power spectral density using the several methods.

   :Parameters:

       **PSD** : :class:`~pioran.psd_base.PowerSpectralDensity`
           Power spectral density object.

       **S_low** : :obj:`float`
           Lower bound of the frequency grid.

       **S_high** : :obj:`float`
           Upper bound of the frequency grid.

       **T** : :obj:`float`
           Duration of the time series.

       **dt** : :obj:`float`
           Minimum sampling duration of the time series.

       **method** : :obj:`str`
           Method used to compute the autocovariance function. Can be 'FFT' if the inverse Fourier transform is used or 'NuFFT'
           for the non uniform Fourier transform. The 'SHO' method will approximate the power spectral density into a sum of SHO functions.

       **n_components** : :obj:`int`
           Number of components used to approximate the power spectral density using the 'SHO' method.

       **estimate_variance** : :obj:`bool`, optional
           If True, the amplitude of the autocovariance function is estimated. Default is True.

       **init_variance** : :obj:`float`, optional
           Initial value of the variance. Default is 1.0.





   :Raises:

       TypeError
           If PSD is not a :class:`~pioran.psd_base.PowerSpectralDensity` object.

       ValueError
           If S_low is smaller than 2., if method is not in the allowed methods or if n_components is smaller than 1.









   ..
       !! processed by numpydoc !!

   .. rubric:: Overview

   .. list-table:: Attributes
      :header-rows: 0
      :widths: auto
      :class: summarytable

      * - :py:obj:`PSD <pioran.PSDToACV.PSD>`
        - Power spectral density object.
      * - :py:obj:`ACVF <pioran.PSDToACV.ACVF>`
        - Autocovariance function as sum of SHO kernels.
      * - :py:obj:`parameters <pioran.PSDToACV.parameters>`
        - Parameters of the power spectral density.
      * - :py:obj:`method <pioran.PSDToACV.method>`
        - Method to compute the covariance function from the power spectral density, by default 'FFT'.Possible values are:
      * - :py:obj:`f_max_obs <pioran.PSDToACV.f_max_obs>`
        - Maximum observed frequency, i.e. the Nyquist frequency.
      * - :py:obj:`f_min_obs <pioran.PSDToACV.f_min_obs>`
        - Minimum observed frequency.
      * - :py:obj:`f0 <pioran.PSDToACV.f0>`
        - Lower bound of the frequency grid.
      * - :py:obj:`S_low <pioran.PSDToACV.S_low>`
        - Scale for the lower bound of the frequency grid.
      * - :py:obj:`S_high <pioran.PSDToACV.S_high>`
        - Scale for the upper bound of the frequency grid.
      * - :py:obj:`fN <pioran.PSDToACV.fN>`
        - Upper bound of the frequency grid.
      * - :py:obj:`estimate_variance <pioran.PSDToACV.estimate_variance>`
        - If True, the amplitude of the autocovariance function is estimated.
      * - :py:obj:`n_freq_grid <pioran.PSDToACV.n_freq_grid>`
        - Number of points in the frequency grid.
      * - :py:obj:`frequencies <pioran.PSDToACV.frequencies>`
        - Frequency grid.
      * - :py:obj:`tau <pioran.PSDToACV.tau>`
        - Time lag grid.
      * - :py:obj:`dtau <pioran.PSDToACV.dtau>`
        - Time lag step.
      * - :py:obj:`n_components <pioran.PSDToACV.n_components>`
        - Number of components used to approximate the power spectral density using the 'SHO' method.
      * - :py:obj:`spectral_points <pioran.PSDToACV.spectral_points>`
        - Frequencies of the SHO kernels.
      * - :py:obj:`spectral_matrix <pioran.PSDToACV.spectral_matrix>`
        - Matrix of the SHO kernels.
      * - :py:obj:`use_celerite <pioran.PSDToACV.use_celerite>`
        - Use celerite2-jax as a backend to model the autocovariance function and compute the log marginal likelihood.
      * - :py:obj:`use_legacy_celerite <pioran.PSDToACV.use_legacy_celerite>`
        - Use celerite2 as a backend to model the autocovariance function and compute the log marginal likelihood.


   .. list-table:: Methods
      :header-rows: 0
      :widths: auto
      :class: summarytable

      * - :py:obj:`decompose_model <pioran.PSDToACV.decompose_model>`\ (psd_normalised)
        - Decompose the PSD model into a sum of basis functions.
      * - :py:obj:`get_approx_coefs <pioran.PSDToACV.get_approx_coefs>`\ ()
        - Get the amplitudes and frequencies of the basis functions.
      * - :py:obj:`build_SHO_model_legacy_cel <pioran.PSDToACV.build_SHO_model_legacy_cel>`\ (amplitudes, frequencies)
        - Build the semi-separable SHO model in celerite from the amplitudes and frequencies.
      * - :py:obj:`build_SHO_model_cel <pioran.PSDToACV.build_SHO_model_cel>`\ (amplitudes, frequencies)
        - Build the semi-separable SHO model in celerite from the amplitudes and frequencies.
      * - :py:obj:`build_DRWCelerite_model_cel <pioran.PSDToACV.build_DRWCelerite_model_cel>`\ (amplitudes, frequencies)
        - Build the semi-separable DRW+Celerite model in celerite from the amplitudes and frequencies.
      * - :py:obj:`build_SHO_model_tinygp <pioran.PSDToACV.build_SHO_model_tinygp>`\ (amplitudes, frequencies)
        - Build the semi-separable SHO model in tinygp from the amplitudes and frequencies.
      * - :py:obj:`calculate <pioran.PSDToACV.calculate>`\ (t, with_ACVF_factor)
        - Calculate the autocovariance function from the power spectral density.
      * - :py:obj:`get_acvf_byNuFFT <pioran.PSDToACV.get_acvf_byNuFFT>`\ (psd, t)
        - Compute the autocovariance function from the power spectral density using the non uniform Fourier transform.
      * - :py:obj:`get_acvf_byFFT <pioran.PSDToACV.get_acvf_byFFT>`\ (psd)
        - Compute the autocovariance function from the power spectral density using the inverse Fourier transform.
      * - :py:obj:`interpolation <pioran.PSDToACV.interpolation>`\ (t, acvf)
        - Interpolate the autocovariance function at the points t.
      * - :py:obj:`get_cov_matrix <pioran.PSDToACV.get_cov_matrix>`\ (xq, xp)
        - Compute the covariance matrix between two arrays xq, xp.
      * - :py:obj:`__str__ <pioran.PSDToACV.__str__>`\ ()
        - String representation of the PSDToACV object.
      * - :py:obj:`__repr__ <pioran.PSDToACV.__repr__>`\ ()
        - Representation of the PSDToACV object.


   .. rubric:: Members

   .. py:attribute:: PSD
      :type: pioran.psd_base.PowerSpectralDensity

      
      Power spectral density object.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: ACVF
      :type: tinygp.kernels.quasisep.SHO

      
      Autocovariance function as sum of SHO kernels.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: parameters
      :type: pioran.parameters.ParametersModel

      
      Parameters of the power spectral density.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: method
      :type: str

      
      Method to compute the covariance function from the power spectral density, by default 'FFT'.Possible values are:
      - 'FFT': use the FFT to compute the autocovariance function.
      - 'NuFFT': use the non-uniform FFT to compute the autocovariance function.
      - 'SHO': approximate the power spectrum as a sum of SHO basis functions to compute the autocovariance function.
      - 'DRWCelerite' : approximate the power spectrum as a sum of DRW+Celerite basis functions to compute the autocovariance function.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: f_max_obs
      :type: float

      
      Maximum observed frequency, i.e. the Nyquist frequency.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: f_min_obs
      :type: float

      
      Minimum observed frequency.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: f0
      :type: float

      
      Lower bound of the frequency grid.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: S_low
      :type: float

      
      Scale for the lower bound of the frequency grid.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: S_high
      :type: float

      
      Scale for the upper bound of the frequency grid.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: fN
      :type: float

      
      Upper bound of the frequency grid.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: estimate_variance
      :type: bool

      
      If True, the amplitude of the autocovariance function is estimated.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: n_freq_grid
      :type: int | None

      
      Number of points in the frequency grid.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: frequencies
      :type: jax.Array | None

      
      Frequency grid.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: tau
      :type: jax.Array
      :value: 0

      
      Time lag grid.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: dtau
      :type: float
      :value: 0

      
      Time lag step.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: n_components
      :type: int
      :value: 0

      
      Number of components used to approximate the power spectral density using the 'SHO' method.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: spectral_points
      :type: jax.Array | None

      
      Frequencies of the SHO kernels.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: spectral_matrix
      :type: jax.Array | None

      
      Matrix of the SHO kernels.
















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

   .. py:method:: decompose_model(psd_normalised: jax.Array)

      
      Decompose the PSD model into a sum of basis functions.

      Assuming that the PSD model can be written as a sum of :math:`J` , this method
      solve the linear system to find the amplitude :math:`a_j` of each kernel.

      .. math:: :label: sho_power_spectrum

      \boldsymbol{y} = B \boldsymbol{a}

      with :math:`\boldsymbol{y}=\begin{bmatrix}1 & \mathcal{P}(f_1)/\mathcal{P}(f_0) & \cdots & \mathcal{P}(f_J)/\mathcal{P}(f_0) \end{bmatrix}^\mathrm{T}`
      the normalised power spectral density vector, :math:`B` the spectral matrix associated to the linear system and :math:`\boldsymbol{a}` the amplitudes of the functions.

      .. math:: :label: sho_spectral_matrix

      B_{ij} = \dfrac{1}{1 + \left(\dfrac{f_i}{f_j}\right)^4}

      .. math:: :label: drwcel_spectral_matrix

      B_{ij} = \dfrac{1}{1 + \left(\dfrac{f_i}{f_j}\right)^6}

      :Parameters:

          **psd_normalised** : :obj:`jax.Array`
              Normalised power spectral density by the first value.

      :Returns:

          :obj:`jax.Array`
              Amplitudes of the functions.

          :obj:`jax.Array`
              Frequencies of the function.













      ..
          !! processed by numpydoc !!

   .. py:method:: get_approx_coefs()

      
      Get the amplitudes and frequencies of the basis functions.

      Estimate the amplitudes and frequencies of the basis functions by solving the linear system.


      :Returns:

          **amplitudes** : :obj:`jax.Array`
              Amplitudes of the SHO kernels.

          **frequencies** : :obj:`jax.Array`
              Frequencies of the SHO kernels.













      ..
          !! processed by numpydoc !!

   .. py:method:: build_SHO_model_legacy_cel(amplitudes: jax.Array, frequencies: jax.Array)

      
      Build the semi-separable SHO model in celerite from the amplitudes and frequencies.

      Currently multiplying the amplitudes to the SHO kernels as sometimes we need negative amplitudes.
      The amplitudes are modelled as a DRW model with c=0.

      :Parameters:

          **amplitudes** : :obj:`jax.Array`
              Amplitudes of the SHO kernels.

          **frequencies** : :obj:`jax.Array`
              Frequencies of the SHO kernels.

      :Returns:

          :obj:`term.Term`
              Constructed SHO kernel.













      ..
          !! processed by numpydoc !!

   .. py:method:: build_SHO_model_cel(amplitudes: jax.Array, frequencies: jax.Array)

      
      Build the semi-separable SHO model in celerite from the amplitudes and frequencies.

      Currently multiplying the amplitudes to the SHO kernels as sometimes we need negative amplitudes.
      The amplitudes are modelled as a DRW model with c=0.

      :Parameters:

          **amplitudes** : :obj:`jax.Array`
              Amplitudes of the SHO kernels.

          **frequencies** : :obj:`jax.Array`
              Frequencies of the SHO kernels.

      :Returns:

          :obj:`term.Term`
              Constructed SHO kernel.













      ..
          !! processed by numpydoc !!

   .. py:method:: build_DRWCelerite_model_cel(amplitudes: jax.Array, frequencies: jax.Array)

      
      Build the semi-separable DRW+Celerite model in celerite from the amplitudes and frequencies.

      The amplitudes

      :Parameters:

          **amplitudes** : :obj:`jax.Array`
              Amplitudes of the SHO kernels.

          **frequencies** : :obj:`jax.Array`
              Frequencies of the SHO kernels.

      :Returns:

          :obj:`term.Term`
              Constructed SHO kernel.













      ..
          !! processed by numpydoc !!

   .. py:method:: build_SHO_model_tinygp(amplitudes: jax.Array, frequencies: jax.Array) -> tinygp.kernels.quasisep.SHO

      
      Build the semi-separable SHO model in tinygp from the amplitudes and frequencies.

      Currently multiplying the amplitudes to the SHO kernels as sometimes we need negative amplitudes,
      which is not possible with the SHO kernel implementation in tinygp.

      :Parameters:

          **amplitudes** : :obj:`jax.Array`
              Amplitudes of the SHO kernels.

          **frequencies** : :obj:`jax.Array`
              Frequencies of the SHO kernels.

      :Returns:

          :obj:`tinygp.kernels.quasisep.SHO`
              Constructed SHO kernel.













      ..
          !! processed by numpydoc !!

   .. py:method:: calculate(t: jax.Array, with_ACVF_factor: bool = False) -> jax.Array

      
      Calculate the autocovariance function from the power spectral density.

      The autocovariance function is computed by the inverse Fourier transform by
      calling the method get_acvf_byFFT. The autocovariance function is then interpolated
      using the method interpolation.

      :Parameters:

          **t** : :obj:`jax.Array`
              Time lags where the autocovariance function is computed.

          **with_ACVF_factor** : :obj:`bool`, optional
              If True, the autocovariance function is multiplied by the factor :math:`\mathcal{R}(0)`. Default is False.

      :Returns:

          :obj:`jax.Array`
              Autocovariance values at the time lags t.




      :Raises:

          NotImplementedError
              If the method is not implemented.









      ..
          !! processed by numpydoc !!

   .. py:method:: get_acvf_byNuFFT(psd: jax.Array, t: jax.Array) -> jax.Array

      
      Compute the autocovariance function from the power spectral density using the non uniform Fourier transform.

      This function uses the jax_finufft package to compute the non uniform Fourier transform with the nufft2 function.

      :Parameters:

          **psd** : :obj:`jax.Array`
              Power spectral density values.

          **t** : :obj:`jax.Array`
              Time lags where the autocovariance function is computed.

      :Returns:

          :obj:`jax.Array`
              Autocovariance values at the time lags t.













      ..
          !! processed by numpydoc !!

   .. py:method:: get_acvf_byFFT(psd: jax.Array) -> jax.Array

      
      Compute the autocovariance function from the power spectral density using the inverse Fourier transform.


      :Parameters:

          **psd** : :obj:`jax.Array`
              Power spectral density.

      :Returns:

          :obj:`jax.Array`
              Autocovariance function.













      ..
          !! processed by numpydoc !!

   .. py:method:: interpolation(t: jax.Array, acvf: jax.Array) -> jax.Array

      
      Interpolate the autocovariance function at the points t.


      :Parameters:

          **t** : :obj:`jax.Array`
              Points where the autocovariance function is computed.

          **acvf** : :obj:`jax.Array`
              Autocovariance values at the points tau.

      :Returns:

          :obj:`jax.Array`
              Autocovariance function at the points t.













      ..
          !! processed by numpydoc !!

   .. py:method:: get_cov_matrix(xq: jax.Array, xp: jax.Array) -> jax.Array

      
      Compute the covariance matrix between two arrays xq, xp.

      The term (xq-xp) is computed using the :func:`~pioran.utils.EuclideanDistance` function from the utils module.
      If the method used is 'NuFFT' and if the two arrays have the same shape, the covariance matrix is computed only on the unique values of the distance matrix
      using the :func:`~pioran.utils.decompose_triangular_matrix` and :func:`~pioran.utils.reconstruct_triangular_matrix` functions from the utils module.
      Otherwise, the covariance matrix is computed on the full distance matrix.

      :Parameters:

          **xq** : :obj:`jax.Array`
              First array.

          **xp** : :obj:`jax.Array`
              Second array.

      :Returns:

          :obj:`jax.Array`
              Covariance matrix.




      :Raises:

          NotImplementedError
              If the method is not implemented.









      ..
          !! processed by numpydoc !!

   .. py:method:: __str__() -> str

      
      String representation of the PSDToACV object.



      :Returns:

          :obj:`str`
              String representation of the PSDToACV object.













      ..
          !! processed by numpydoc !!

   .. py:method:: __repr__() -> str

      
      Representation of the PSDToACV object.
















      ..
          !! processed by numpydoc !!



.. py:class:: Inference(Process: pioran.core.GaussianProcess | pioran.carma.carma_core.CARMAProcess, priors, method, n_samples_checks=1000, seed_check=0, run_checks=True, log_dir='log_dir', title_plots=True)

   
   Class to infer the value of the (hyper)parameters of the Gaussian Process.

   Various methods to sample the posterior probability distribution of the (hyper)parameters of the Gaussian Process are implemented
   as wrappers around the inference packages `blackjax` and `ultranest`.

   :Parameters:

       **Process** : :class:`~pioran.core.GaussianProcess`
           Process object.

       **priors** : :obj:`function`
           Function to define the priors for the (hyper)parameters.

       **method** : :obj:`str`, optional
           "NS": using nested sampling via ultranest

       **n_samples_checks** : :obj:`int`, optional
           Number of samples to take from the prior distribution, by default 1000

       **seed_check** : :obj:`int`, optional
           Seed for the random number generator, by default 0

       **run_checks** : :obj:`bool`, optional
           Run the prior predictive checks, by default True

       **log_dir** : :obj:`str`, optional
           Directory to save the results of the inference, by default 'log_dir'

       **title_plots** : :obj:`bool`, optional
           Plot the title of the figures, by default True





   :Raises:

       ImportError
           If the required packages are not installed.

       ValueError
           If the saved config file is different from the current config, or if the method is not valid.

       TypeError
           If the method is not a string.









   ..
       !! processed by numpydoc !!

   .. rubric:: Overview

   .. list-table:: Attributes
      :header-rows: 0
      :widths: auto
      :class: summarytable

      * - :py:obj:`process <pioran.Inference.process>`
        - Process object.
      * - :py:obj:`n_pars <pioran.Inference.n_pars>`
        - Number of (hyper)parameters.
      * - :py:obj:`priors <pioran.Inference.priors>`
        - Function to define the priors for the (hyper)parameters.
      * - :py:obj:`log_dir <pioran.Inference.log_dir>`
        - Directory to save the results of the inference.
      * - :py:obj:`plot_dir <pioran.Inference.plot_dir>`
        - Directory to save the plots of the inference.
      * - :py:obj:`method <pioran.Inference.method>`
        - Method to use for the inference.


   .. list-table:: Methods
      :header-rows: 0
      :widths: auto
      :class: summarytable

      * - :py:obj:`save_config <pioran.Inference.save_config>`\ (save_file)
        - Save the configuration of the inference.
      * - :py:obj:`prior_predictive_checks <pioran.Inference.prior_predictive_checks>`\ (n_samples_checks, seed_check, n_frequencies, plot_prior_samples, plot_prior_predictive_distribution)
        - Check the prior predictive distribution.
      * - :py:obj:`check_approximation <pioran.Inference.check_approximation>`\ (n_samples_checks, seed_check, n_frequencies, plot_diagnostics, plot_violins, plot_quantiles, title)
        - Check the approximation of the PSD with the kernel decomposition.
      * - :py:obj:`run <pioran.Inference.run>`\ (verbose, user_log_likelihood, seed, n_chains, n_samples, n_warmup_steps, use_stepsampler)
        - Estimate the (hyper)parameters of the Gaussian Process.
      * - :py:obj:`blackjax_DYHMC <pioran.Inference.blackjax_DYHMC>`\ (rng_key, initial_position, log_likelihood, log_prior, num_warmup_steps, num_samples, num_chains, step_size, learning_rate)
        - Sample the posterior distribution using the NUTS sampler from blackjax.
      * - :py:obj:`blackjax_NUTS <pioran.Inference.blackjax_NUTS>`\ (rng_key, initial_position, log_likelihood, log_prior, num_warmup_steps, num_samples, num_chains)
        - Sample the posterior distribution using the NUTS sampler from blackjax.
      * - :py:obj:`nested_sampling <pioran.Inference.nested_sampling>`\ (priors, log_likelihood, verbose, use_stepsampler, resume, run_kwargs, slice_steps)
        - Sample the posterior distribution of the (hyper)parameters of the Gaussian Process with nested sampling via ultranest.


   .. rubric:: Members

   .. py:attribute:: process
      :type: pioran.core.GaussianProcess | pioran.carma.carma_core.CARMAProcess

      
      Process object.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: n_pars
      :type: int

      
      Number of (hyper)parameters.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: priors
      :type: callable

      
      Function to define the priors for the (hyper)parameters.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: log_dir
      :type: str

      
      Directory to save the results of the inference.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: plot_dir
      :type: str

      
      Directory to save the plots of the inference.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: method
      :type: str

      
      Method to use for the inference.
















      ..
          !! processed by numpydoc !!

   .. py:method:: save_config(save_file=True)

      
      Save the configuration of the inference.

      Save the configuration of the inference, process and model in a json file.

      :Parameters:

          **save_file** : :obj:`bool`, optional
              ..

      :Returns:

          **dict_config** : :obj:`dict`
              Dictionary with the configuration of the inference, process and model.













      ..
          !! processed by numpydoc !!

   .. py:method:: prior_predictive_checks(n_samples_checks, seed_check, n_frequencies=1000, plot_prior_samples=True, plot_prior_predictive_distribution=True)

      
      Check the prior predictive distribution.

      Get samples from the prior distribution and plot them, and calculate the prior predictive
      distribution of the model and plot it.

      :Parameters:

          **n_samples_checks** : :obj:`int`
              Number of samples to take from the prior distribution, by default 1000

          **seed_check** : :obj:`int`
              Seed for the random number generator

          **plot_prior_samples** : :obj:`bool`, optional
              Plot the prior samples, by default True

          **plot_prior_predictive_distributions** : :obj:`bool`, optional
              Plot the prior predictive distribution of the model, by default True














      ..
          !! processed by numpydoc !!

   .. py:method:: check_approximation(n_samples_checks: int, seed_check: int, n_frequencies: int = 1000, plot_diagnostics: bool = True, plot_violins: bool = True, plot_quantiles: bool = True, title: bool = True)

      
      Check the approximation of the PSD with the kernel decomposition.

      This method will take random samples from the prior distribution and compare the PSD obtained
      with the SHO decomposition with the true PSD.

      :Parameters:

          **n_samples_checks** : :obj:`int`
              Number of samples to take from the prior distribution, by default 1000

          **seed_check** : :obj:`int`
              Seed for the random number generator

          **n_frequencies** : :obj:`int`, optional
              Number of frequencies to evaluate the PSD, by default 1000

          **plot_diagnostics** : :obj:`bool`, optional
              Plot the diagnostics of the approximation, by default True

          **plot_violins** : :obj:`bool`, optional
              Plot the violin plots of the residuals and the ratios, by default True

          **plot_quantiles** : :obj:`bool`, optional
              Plot the quantiles of the residuals and the ratios, by default True

          **plot_prior_samples** : :obj:`bool`, optional
              Plot the prior samples, by default True

          **title** : :obj:`bool`, optional
              Plot the title of the figure, by default True

      :Returns:

          **figs** : :obj:`list`
              List of figures.

          **residuals** : :obj:`jax.Array`
              Residuals of the PSD approximation.

          **ratio** : :obj:`jax.Array`
              Ratio of the PSD approximation.













      ..
          !! processed by numpydoc !!

   .. py:method:: run(verbose: bool = True, user_log_likelihood=None, seed: int = 0, n_chains: int = 1, n_samples: int = 1000, n_warmup_steps: int = 1000, use_stepsampler: bool = False)

      
      Estimate the (hyper)parameters of the Gaussian Process.

      Run the inference method.

      :Parameters:

          **verbose** : :obj:`bool`, optional
              Be verbose, by default True

          **user_log_likelihood** : :obj:`function`, optional
              User-defined function to compute the log-likelihood, by default None

          **seed** : :obj:`int`, optional
              Seed for the random number generator, by default 0

          **n_chains** : :obj:`int`, optional
              Number of chains, by default 1

          **n_samples** : :obj:`int`, optional
              Number of samples to take from the posterior distribution, by default 1_000

          **n_warmup_steps** : :obj:`int`, optional
              Number of warmup steps, by default 1_000

          **use_stepsampler** : :obj:`bool`, optional
              Use the slice sampler as step sampler, by default False

      :Returns:

          results: dict
              Results of the sampling. The keys differ depending on the method/sampler used.













      ..
          !! processed by numpydoc !!

   .. py:method:: blackjax_DYHMC(rng_key: jax.random.PRNGKey, initial_position: jax.Array, log_likelihood: callable, log_prior: callable, num_warmup_steps: int = 1000, num_samples: int = 1000, num_chains: int = 1, step_size: float = 0.01, learning_rate: float = 0.01)

      
      Sample the posterior distribution using the NUTS sampler from blackjax.

      Wrapper around the NUTS sampler from blackjax to sample the posterior distribution.
      This function also performs the warmup via window adaptation.

      :Parameters:

          **rng_key** : :obj:`jax.random.PRNGKey`
              Random key for the random number generator.

          **initial_position** : :obj:`jax.Array`
              Initial position of the chains.

          **log_likelihood** : :obj:`function`
              Function to compute the log-likelihood.

          **log_prior** : :obj:`function`
              Function to compute the log-prior.

          **num_warmup_steps** : :obj:`int`, optional
              Number of warmup steps, by default 1_000

          **num_samples** : :obj:`int`, optional
              Number of samples to take from the posterior distribution, by default 1_000

          **num_chains** : :obj:`int`, optional
              Number of chains, by default 1

      :Returns:

          **samples** : :obj:`jax.Array`
              Samples from the posterior distribution. It has shape (num_chains, num_params, num_samples).

          **log_prob** : :obj:`jax.Array`
              Log-probability of the samples.













      ..
          !! processed by numpydoc !!

   .. py:method:: blackjax_NUTS(rng_key: jax.random.PRNGKey, initial_position: jax.Array, log_likelihood: callable, log_prior: callable, num_warmup_steps: int = 1000, num_samples: int = 1000, num_chains: int = 1)

      
      Sample the posterior distribution using the NUTS sampler from blackjax.

      Wrapper around the NUTS sampler from blackjax to sample the posterior distribution.
      This function also performs the warmup via window adaptation.

      :Parameters:

          **rng_key** : :obj:`jax.random.PRNGKey`
              Random key for the random number generator.

          **initial_position** : :obj:`jax.Array`
              Initial position of the chains.

          **log_likelihood** : :obj:`function`
              Function to compute the log-likelihood.

          **log_prior** : :obj:`function`
              Function to compute the log-prior.

          **num_warmup_steps** : :obj:`int`, optional
              Number of warmup steps, by default 1_000

          **num_samples** : :obj:`int`, optional
              Number of samples to take from the posterior distribution, by default 1_000

          **num_chains** : :obj:`int`, optional
              Number of chains, by default 1

      :Returns:

          **samples** : :obj:`jax.Array`
              Samples from the posterior distribution. It has shape (num_chains, num_params, num_samples).

          **log_prob** : :obj:`jax.Array`
              Log-probability of the samples.













      ..
          !! processed by numpydoc !!

   .. py:method:: nested_sampling(priors: callable, log_likelihood: callable, verbose: bool = True, use_stepsampler: bool = False, resume: bool = True, run_kwargs={}, slice_steps=100)

      
      Sample the posterior distribution of the (hyper)parameters of the Gaussian Process with nested sampling via ultranest.

      Perform nested sampling to sample the (hyper)parameters of the Gaussian Process.

      :Parameters:

          **priors** : :obj:`function`
              Function to define the priors for the parameters

          **log_likelihood** : :obj:`function`
              Function to compute the log-likelihood.

          **verbose** : :obj:`bool`, optional
              Print the results of the sample and the progress of the sampling, by default True

          **use_stepsampler** : :obj:`bool`, optional
              Use the slice sampler as step sampler, by default False

          **resume** : :obj:`bool`, optional
              Resume the sampling from the previous run, by default True

          **run_kwargs** : :obj:`dict`, optional
              Dictionary of arguments for ReactiveNestedSampler.run() see https://johannesbuchner.github.io/UltraNest/ultranest.html#module-ultranest.integrator

          **slice_steps** : :obj:`int`, optional
              Number of steps for the slice sampler, by default 100

      :Returns:

          results: dict
              Dictionary of results from the nested sampling.













      ..
          !! processed by numpydoc !!



.. py:class:: Simulations(T, dt, model: pioran.psd_base.PowerSpectralDensity | pioran.acvf_base.CovarianceFunction, N=None, S_low=None, S_high=None)

   
   Simulate time series from a given PSD or ACVF model.


   :Parameters:

       **T** : :obj:`float`
           duration of the time series.

       **dt** : :obj:`float`
           sampling period of the time series.

       **model** : :class:`~pioran.acvf_base.CovarianceFunction` or :class:`~pioran.psd_base.PowerSpectralDensity`
           The model for the simulation of the process, can be a PSD or an ACVF.

       **S_low** : :obj:`float`, optional
           Scale factor for the lower bound of the frequency grid.
           If the model is a PSD, this parameter is mandatory.

       **S_high** : :obj:`float`, optional
           Scale factor for the upper bound of the frequency grid.
           If the model is a PSD, this parameter is mandatory.





   :Raises:

       ValueError
           If the model is not a PSD or ACVF.









   ..
       !! processed by numpydoc !!

   .. rubric:: Overview

   .. list-table:: Attributes
      :header-rows: 0
      :widths: auto
      :class: summarytable

      * - :py:obj:`duration <pioran.Simulations.duration>`
        - duration of the time series.
      * - :py:obj:`sampling_period <pioran.Simulations.sampling_period>`
        - sampling period of the time series.
      * - :py:obj:`n_time <pioran.Simulations.n_time>`
        - number of time indexes.
      * - :py:obj:`t <pioran.Simulations.t>`
        - time :obj:`jax.Array` of the time series.
      * - :py:obj:`f_max_obs <pioran.Simulations.f_max_obs>`
        - maximum frequency of the observed frequency grid.
      * - :py:obj:`f_min_obs <pioran.Simulations.f_min_obs>`
        - minimum frequency of the observed frequency grid.
      * - :py:obj:`f0 <pioran.Simulations.f0>`
        - minimum frequency of the total frequency grid.
      * - :py:obj:`fN <pioran.Simulations.fN>`
        - maximum frequency of the total frequency grid.
      * - :py:obj:`n_freq_grid <pioran.Simulations.n_freq_grid>`
        - number of frequency indexes.
      * - :py:obj:`frequencies <pioran.Simulations.frequencies>`
        - frequency array of the total frequency grid.
      * - :py:obj:`tau_max <pioran.Simulations.tau_max>`
        - maximum lag of the autocovariance function.
      * - :py:obj:`dtau <pioran.Simulations.dtau>`
        - sampling period of the autocovariance function.
      * - :py:obj:`tau <pioran.Simulations.tau>`
        - lag array of the autocovariance function.
      * - :py:obj:`psd <pioran.Simulations.psd>`
        - power spectral density of the time series.
      * - :py:obj:`acvf <pioran.Simulations.acvf>`
        - autocovariance function of the time series.
      * - :py:obj:`triang <pioran.Simulations.triang>`
        - triangular matrix used to generate the time series with the Cholesky decomposition.
      * - :py:obj:`keys <pioran.Simulations.keys>`
        - dictionary of the keys used to generate the random numbers. See :func:`~pioran.simulate.Simulations.generate_keys` for more details.


   .. list-table:: Methods
      :header-rows: 0
      :widths: auto
      :class: summarytable

      * - :py:obj:`generate_keys <pioran.Simulations.generate_keys>`\ (seed)
        - Generate the keys to generate the random numbers.
      * - :py:obj:`plot_acvf <pioran.Simulations.plot_acvf>`\ (figsize, xunit, filename, title)
        - Plot the autocovariance function.
      * - :py:obj:`plot_psd <pioran.Simulations.plot_psd>`\ (figsize, filename, title, xunit, loglog)
        - Plot the power spectral density model.
      * - :py:obj:`GP_method <pioran.Simulations.GP_method>`\ (t_test, interpolation)
        - Generate a time series using the GP method.
      * - :py:obj:`simulate <pioran.Simulations.simulate>`\ (mean, method, irregular_sampling, irregular_gaps, randomise_fluxes, errors, errors_size, N_points, seed, seed_gaps, filename, exponentiate_ts, min_n_gaps, max_n_gaps, max_size_slices, interp_method)
        - Method to simulate time series using either the GP method or the TK method.
      * - :py:obj:`simulate_irregular_gaps <pioran.Simulations.simulate_irregular_gaps>`\ (a, seed, N_points, min_n_gaps, max_n_gaps, max_size_slices)
        - Simulate irregular times from a regular time series with random gaps.
      * - :py:obj:`extract_subset_timeseries <pioran.Simulations.extract_subset_timeseries>`\ (t, y, M)
        - Select a random subset of points from an input time series.
      * - :py:obj:`sample_timeseries <pioran.Simulations.sample_timeseries>`\ (t, y, M, irregular_sampling)
        - Extract a random subset of points from the time series.
      * - :py:obj:`timmer_Koenig_method <pioran.Simulations.timmer_Koenig_method>`\ ()
        - Generate a time series using the Timmer-Konig method.
      * - :py:obj:`split_longtimeseries <pioran.Simulations.split_longtimeseries>`\ (t, ts, n_slices)
        - Split a long time series into shorter time series.
      * - :py:obj:`resample_longtimeseries <pioran.Simulations.resample_longtimeseries>`\ (t_slices, ts_slices)
        - Resample the time series to have a regular sampling period with n_time points.
      * - :py:obj:`simulate_longtimeseries <pioran.Simulations.simulate_longtimeseries>`\ (mean, randomise_fluxes, errors, seed)
        - Method to simulate several long time series using the Timmer-Koenig method.


   .. rubric:: Members

   .. py:attribute:: duration
      :type: float

      
      duration of the time series.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: sampling_period
      :type: float

      
      sampling period of the time series.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: n_time
      :type: int

      
      number of time indexes.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: t
      :type: jax.Array

      
      time :obj:`jax.Array` of the time series.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: f_max_obs
      :type: float

      
      maximum frequency of the observed frequency grid.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: f_min_obs
      :type: float

      
      minimum frequency of the observed frequency grid.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: f0
      :type: float

      
      minimum frequency of the total frequency grid.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: fN
      :type: float

      
      maximum frequency of the total frequency grid.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: n_freq_grid
      :type: int

      
      number of frequency indexes.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: frequencies
      :type: jax.Array

      
      frequency array of the total frequency grid.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: tau_max
      :type: float

      
      maximum lag of the autocovariance function.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: dtau
      :type: float

      
      sampling period of the autocovariance function.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: tau
      :type: jax.Array

      
      lag array of the autocovariance function.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: psd
      :type: jax.Array

      
      power spectral density of the time series.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: acvf
      :type: jax.Array

      
      autocovariance function of the time series.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: triang
      :type: bool

      
      triangular matrix used to generate the time series with the Cholesky decomposition.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: keys
      :type: dict

      
      dictionary of the keys used to generate the random numbers. See :func:`~pioran.simulate.Simulations.generate_keys` for more details.
















      ..
          !! processed by numpydoc !!

   .. py:method:: generate_keys(seed: int = 0) -> None

      
      Generate the keys to generate the random numbers.

      This function generates the keys to generate the random numbers for the simulations and store them in the dictionary self.keys.
      The keys and their meaning are:

      - `simu_TS`  : key for drawing the values of the time series.
      - `errors`   : key for drawing the size of the errorbar of the time series from a given distribution.
      - `fluxes`   : key for drawing the fluxes of the time series from a given distribution.
      - `subset`   : key for randomising the choice of the subset of the time series.
      - `sampling` : key for randomising the choice of the sampling of the time series.

      :Parameters:

          **seed** : :obj:`int`, optional
              Seed for the random number generator, by default 0














      ..
          !! processed by numpydoc !!

   .. py:method:: plot_acvf(figsize=(9, 5.5), xunit='d', filename=None, title=None)

      
      Plot the autocovariance function.

      Plot the autocovariance function of the time series.

      :Parameters:

          **figsize** : :obj:`tuple`, optional
              Size of the figure, by default (15,3)

          **xunit** : :obj:`str`, optional
              Unit of the x-axis, by default 'd'

          **filename** : :obj:`str`, optional
              Name of the file to save the figure, by default None

          **title** : :obj:`str`, optional
              Title of the plot, by default None

      :Returns:

          :obj:`matplotlib.figure.Figure`
                 Figure of the plot
              :obj:`matplotlib.axes.Axes`
                 Axes of the plot













      ..
          !! processed by numpydoc !!

   .. py:method:: plot_psd(figsize=(6, 4), filename=None, title=None, xunit='d', loglog=True)

      
      Plot the power spectral density model.

      A plot of the power spectral density model is generated.

      :Parameters:

          **figsize** : :obj:`tuple`, optional
              Size of the figure, by default (6,4)

          **filename** : :obj:`str`, optional
              Name of the file to save the figure, by default None

          **title** : :obj:`str`, optional
              Title of the plot, by default None

          **xunit** : :obj:`str`, optional
              Unit of the x-axis, by default 'd'

          **loglog** : :obj:`bool`, optional
              If True, the plot is in loglog, by default True

      :Returns:

          :obj:`matplotlib.figure.Figure`
              Figure of the plot

          :obj:`matplotlib.axes.Axes`
              Axes of the plot













      ..
          !! processed by numpydoc !!

   .. py:method:: GP_method(t_test: jax.Array, interpolation='cubic') -> tuple[jax.Array, jax.Array]

      
      Generate a time series using the GP method.

      If the ACVF is not already calculated, it is calculated from the PSD
      using the inverse Fourier transform.

      :Parameters:

          **t_test: :obj:`jax.Array`**
              Time array of the time series.

          **interpolation** : :obj:`str`, optional
              Interpolation method to use for the GP function, by default 'cubic'.

      :Returns:

          :obj:`jax.Array`
              Time array of the time series.

          :obj:`jax.Array`
              Time series.




      :Raises:

          ValueError
              If the interpolation method is not 'linear' or 'cubic'.









      ..
          !! processed by numpydoc !!

   .. py:method:: simulate(mean: float | str | None = 'default', method: str = 'GP', irregular_sampling: bool = False, irregular_gaps: bool = False, randomise_fluxes: bool = True, errors: str = 'gauss', errors_size: float = 0.02, N_points: int = 0, seed: int = 0, seed_gaps: int = 1, filename: str = '', exponentiate_ts: bool = False, min_n_gaps: int = 0, max_n_gaps: int = 10, max_size_slices: float = 2.0, interp_method: str = 'cubic') -> tuple[jax.Array, jax.Array, jax.Array]

      
      Method to simulate time series using either the GP method or the TK method.

      When using the GP method, the time series is generated using an analytical autocovariance function or a power spectral density.
      If the autocovariance function is not provided, it is calculated from the power spectral density using the inverse Fourier transform
      and interpolated using a linear interpolation to map the autocovariance function on a grid of time lags.

      When using the TK method, the time series is generated using the :func:`~pioran.simulate.Simulations.timmer_Koenig_method` method for a larger duration and then the final time series
      is obtained by taking a subset of the generate time series.

      If irregular_sampling is set to `True`, the time series will be sampled at random irregular time intervals.

      :Parameters:

          **mean** : :obj:`float` or :obj:`str`, optional
              Mean of the time series, 'default' sets the mean to 0 for non-exponentiated, if None the mean will be set to -2 min(ts)

          **method** : :obj:`str`, optional
              method to simulate the time series, by default 'GP'
              can be 'TK' which uses Timmer and Koening method

          **randomise_fluxes** : :obj:`bool`, optional
              If True the fluxes will be randomised.

          **errors** : :obj:`str`, optional
              If 'gauss' the errors will be drawn from a gaussian distribution

          **errors_size** : :obj:`float`, optional
              Size of the errors on the time series, by default 0.02 (2%)

          **N_points** : :obj:`int`, optional
              Number of points to sample when simulating irregular gaps, by default 0

          **irregular_sampling** : :obj:`bool`, optional
              If True the time series will be sampled at irregular time intervals

          **irregular_gaps** : :obj:`bool`, optional
              If True the time series will be sampled at irregular time intervals with random gaps

          **seed** : :obj:`int`, optional
              Set the seed for the RNG

          **seed_gaps** : :obj:`int`, optional
              Set the seed for the RNG when simulating irregular gaps

          **exponentiate_ts: :obj:`bool`, optional**
              Exponentiate the time series to produce a lognormal flux distribution.

          **filename** : :obj:`str`, optional
              Name of the file to save the time series, by default ''

          **min_n_gaps** : :obj:`int`, optional
              Minimum number of gaps when simulating irregular gaps, by default 2

          **max_n_gaps** : :obj:`int`, optional
              Maximum number of gaps when simulating irregular gaps, by default 22

          **max_size_slices: obj:`float`**
              Max size factor, default it 2.

          **interp_method** : :obj:`str`, optional
              Interpolation method to use when calculating the autocovariance function from the power spectral density, by default 'linear'

          **Raises**
              ..

          **------**
              ..

          **ValueError**
              If the method is not 'GP' or 'TK'

          **ValueError**
              If the errors are not 'gauss' or 'poisson'

      :Returns:

          **t** : :obj:`jax.Array`
              The time indexes of the time series.

          **ts** : :obj:`jax.Array`
              Values of the simulated time series.

          **ts_err** : :obj:`jax.Array`
              Errors on the simulated time series













      ..
          !! processed by numpydoc !!

   .. py:method:: simulate_irregular_gaps(a: jax.Array, seed: int, N_points: int, min_n_gaps: int = 2, max_n_gaps: int = 22, max_size_slices: float = 2.0)

      
      Simulate irregular times from a regular time series with random gaps.


      :Parameters:

          **a** : :obj:`jax.Array`
              Regular time series indexes.

          **key** : :obj:`jax.random.PRNGKey`
              Random key.

          **N_points** : :obj:`int`
              Number of points to sample.

          **min_n_gaps** : :obj:`int`
              Minimum number of gaps. Default is 2.

          **max_n_gaps** : :obj:`int`
              Maximum number of gaps. Default is 22.

          **max_size_slices** : obj:`float`
              Max size factor, default it 2.

      :Returns:

          :obj:`jax.Array`
              Irregular times.

          :obj:`jax.Array`
              Indexes of the irregular times.













      ..
          !! processed by numpydoc !!

   .. py:method:: extract_subset_timeseries(t: jax.Array, y: jax.Array, M: int) -> tuple[jax.Array, jax.Array]

      
      Select a random subset of points from an input time series.

      The input time series is regularly sampled of size N.
      The output time series is of size M with the same sampling rate as the input time series.

      :Parameters:

          **t** : :obj:`jax.Array`
              Input time series of size N.

          **y** : :obj:`jax.Array`
              The fluxes of the simulated light curve.

          **M** : :obj:`int`
              The number of points in the desired time series.

      :Returns:

          :obj:`jax.Array`
              The time series of size M.

          :obj:`jax.Array`
              The values of the time series of size M.













      ..
          !! processed by numpydoc !!

   .. py:method:: sample_timeseries(t: jax.Array, y: jax.Array, M: int, irregular_sampling: bool = False)

      
      Extract a random subset of points from the time series.

      Extract a random subset of M points from the time series. The input time series t is regularly sampled of size N with a sampling period dT.
      If irregular_sampling is False, the output time series has a sampling period dT/M.
      If irregular_sampling is True, the output time series is irregularly sampled.

      :Parameters:

          **t** : :obj:`jax.Array`
              The time indexes of the time series.

          **y** : :obj:`jax.Array`
              The values of the time series.

          **M** : :obj:`int`
              The number of points in the desired time series.

          **irregular_sampling** : :obj:`bool`
              If True, the time series is irregularly sampled. If False, the time series is regularly sampled.

      :Returns:

          :obj:`jax.Array`
              The time indexes of the sampled time series.

          :obj:`jax.Array`
              The values of the sampled time series.













      ..
          !! processed by numpydoc !!

   .. py:method:: timmer_Koenig_method() -> tuple[jax.Array, jax.Array]

      
      Generate a time series using the Timmer-Konig method.

      Use the Timmer-Konig method to generate a time series with a given power spectral density
      stored in the attribute psd.

      Assuming a power-law shaped PSD, the method is as follows:

      Draw two independent Gaussian random variables N1 and N2 with zero mean and unit variance.
      The random variables are drawn using the key self.keys['ts'] split into two subkeys.

          1. Define A = sqrt(PSD/2) * (N1 + i*N2)
          2. Define A[0] = 0
          3. Define A[-1] = real(A[-1])
          4. ts = irfft(A)
          5. t is defined as the time indexes of the time series, with a sampling period of 0.5/fN.
          6. ts is multiplied by the 2*len(psd)*sqrt(f0) factor to ensure that the time series has the correct variance.

      The duration of the output time series is 2*(len(psd)-1).


      :Returns:

          :obj:`jax.Array`
              The time indexes of the time series.

          :obj:`jax.Array`
              The values of the time series.













      ..
          !! processed by numpydoc !!

   .. py:method:: split_longtimeseries(t: jax.Array, ts: jax.Array, n_slices: int) -> tuple[list, list]

      
      Split a long time series into shorter time series.

      Break the time series into n_slices shorter time series. The short time series are of equal length.

      :Parameters:

          **t** : :obj:`jax.Array`
              The time indexes of the long time series.

          **ts** : :obj:`jax.Array`
              The values of the long time series.

          **n_slices** : :obj:`int`
              The number of slices to break the time series into.

      :Returns:

          :obj:`list`
              A list of the time indexes of the shorter time series.

          :obj:`list`
              A list of the values of the shorter time series.













      ..
          !! processed by numpydoc !!

   .. py:method:: resample_longtimeseries(t_slices: list, ts_slices: list) -> tuple[list, list]

      
      Resample the time series to have a regular sampling period with n_time points.


      :Parameters:

          **t_slices** : :obj:`list`
              A list of short time series time indexes.

          **ts_slices** : :obj:`list`
              A list of short time series values.

      :Returns:

          :obj:`list`
              A list of the time indexes of the sampled time series.

          :obj:`list`
              A list of the values of the sampled time series.













      ..
          !! processed by numpydoc !!

   .. py:method:: simulate_longtimeseries(mean: float | None = None, randomise_fluxes: bool = True, errors: str = 'gauss', seed: int = 0) -> tuple[list, list, list]

      
      Method to simulate several long time series using the Timmer-Koenig method.

      The time series is generated using the :func:`~pioran.simulate.Simulations.timmer_Koenig_method` method for a larger duration and then the final time series
      are split into segments of length n_time. The shorter time series are then resampled to have a regular sampling period.

      :Parameters:

          **mean** : :obj:`float`, optional
              Mean of the time series, if None the mean will be set to -2 min(ts)

          **randomise_fluxes** : :obj:`bool`, optional
              If True the fluxes will be randomised.

          **errors** : :obj:`str`, optional
              If 'gauss' the errors will be drawn from a gaussian distribution

          **seed** : :obj:`int`, optional
              Set the seed for the RNG

      :Returns:

          :obj:`list`
              A list of the time indexes of the segments.

          :obj:`list`
              A list of the values of the segments.

          :obj:`list`
              A list of the errors of the segments.




      :Raises:

          ValueError
              If the errors are not 'gauss' or 'poisson'









      ..
          !! processed by numpydoc !!



.. py:class:: Visualisations(process: pioran.core.GaussianProcess | pioran.carma.carma_core.CARMAProcess, filename: str, n_frequencies: int = 2500)

   
   Class for visualising the results after an inference run.


   :Parameters:

       **process** : :obj:`GaussianProcess` or :obj:`CARMAProcess`
           The process to be visualised.

       **filename** : :obj:`str`
           The filename prefix for the output plots.

       **n_frequencies** : :obj:`int`, optional
           The number of frequencies at which to evaluate the PSDs, by default 2500.














   ..
       !! processed by numpydoc !!

   .. rubric:: Overview

   .. list-table:: Attributes
      :header-rows: 0
      :widths: auto
      :class: summarytable

      * - :py:obj:`process <pioran.Visualisations.process>`
        - The process to be visualised.
      * - :py:obj:`x <pioran.Visualisations.x>`
        - The observation times.
      * - :py:obj:`y <pioran.Visualisations.y>`
        - The observation values.
      * - :py:obj:`yerr <pioran.Visualisations.yerr>`
        - The observation errors.
      * - :py:obj:`predictive_mean <pioran.Visualisations.predictive_mean>`
        - The predictive mean.
      * - :py:obj:`predictive_cov <pioran.Visualisations.predictive_cov>`
        - The predictive covariance.
      * - :py:obj:`x_pred <pioran.Visualisations.x_pred>`
        - The prediction times.
      * - :py:obj:`f_min <pioran.Visualisations.f_min>`
        - The minimum frequency.
      * - :py:obj:`f_max <pioran.Visualisations.f_max>`
        - The maximum frequency.
      * - :py:obj:`frequencies <pioran.Visualisations.frequencies>`
        - The frequencies at which to evaluate the PSDs.
      * - :py:obj:`tau <pioran.Visualisations.tau>`
        - The times at which to evaluate the ACFs.
      * - :py:obj:`filename_prefix <pioran.Visualisations.filename_prefix>`
        - The filename prefix for the output plots.
      * - :py:obj:`process_legacy <pioran.Visualisations.process_legacy>`
        - \-


   .. list-table:: Methods
      :header-rows: 0
      :widths: auto
      :class: summarytable

      * - :py:obj:`plot_timeseries_diagnostics <pioran.Visualisations.plot_timeseries_diagnostics>`\ (samples, prediction_indexes, n_samples)
        - Plot the timeseries diagnostics using samples from the posterior distribution.
      * - :py:obj:`plot_timeseries_diagnostics_old <pioran.Visualisations.plot_timeseries_diagnostics_old>`\ (prediction_indexes, \*\*kwargs)
        - Plot the timeseries diagnostics.
      * - :py:obj:`posterior_predictive_checks <pioran.Visualisations.posterior_predictive_checks>`\ (samples, plot_PSD, plot_ACVF, \*\*kwargs)
        - Plot the posterior predictive checks.


   .. rubric:: Members

   .. py:attribute:: process
      :type: pioran.core.GaussianProcess | pioran.carma.carma_core.CARMAProcess

      
      The process to be visualised.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: x
      :type: jax.Array

      
      The observation times.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: y
      :type: jax.Array

      
      The observation values.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: yerr
      :type: jax.Array

      
      The observation errors.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: predictive_mean
      :type: jax.Array

      
      The predictive mean.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: predictive_cov
      :type: jax.Array

      
      The predictive covariance.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: x_pred
      :type: jax.Array

      
      The prediction times.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: f_min
      :type: float

      
      The minimum frequency.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: f_max
      :type: float

      
      The maximum frequency.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: frequencies
      :type: jax.Array

      
      The frequencies at which to evaluate the PSDs.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: tau
      :type: jax.Array

      
      The times at which to evaluate the ACFs.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: filename_prefix
      :type: str

      
      The filename prefix for the output plots.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: process_legacy
      :type: pioran.core.GaussianProcess

      

   .. py:method:: plot_timeseries_diagnostics(samples, prediction_indexes: jax.Array | None = None, n_samples: int = 400) -> None

      
      Plot the timeseries diagnostics using samples from the posterior distribution.

      This function will call the :func:`plot_prediction` and :func:`plot_residuals` functions to
      plot the predicted timeseries and the residuals.

      :Parameters:

          **samples** : :obj:`jax.Array`
              The samples from the posterior distribution.

          **prediction_indexes** : :obj:`jax.Array`, optional
              The prediction times, by default None

          **n_samples** : :obj:`int`, optional
              The number of samples to use for the posterior predictive checks, by default 400

          **\*\*kwargs**
              Additional keyword arguments to be passed to the :func:`plot_prediction` and :func:`plot_residuals` functions.














      ..
          !! processed by numpydoc !!

   .. py:method:: plot_timeseries_diagnostics_old(prediction_indexes: jax.Array | None = None, **kwargs) -> None

      
      Plot the timeseries diagnostics.

      This function will call the :func:`plot_prediction` and :func:`plot_residuals` functions to
      plot the predicted timeseries and the residuals.

      :Parameters:

          **prediction_indexes** : :obj:`jax.Array`, optional
              The prediction times, by default None

          **\*\*kwargs**
              Additional keyword arguments to be passed to the :func:`plot_prediction` and :func:`plot_residuals` functions.














      ..
          !! processed by numpydoc !!

   .. py:method:: posterior_predictive_checks(samples: jax.Array, plot_PSD: bool = True, plot_ACVF: bool = True, **kwargs)

      
      Plot the posterior predictive checks.


      :Parameters:

          **samples** : :obj:`jax.Array`
              The samples from the posterior distribution.

          **plot_PSD** : :obj:`bool`, optional
              Plot the posterior predictive PSDs, by default True

          **plot_ACVF** : :obj:`bool`, optional
              Plot the posterior predictive ACVFs, by default True

          **\*\*kwargs**
              Additional keyword arguments.
              frequencies : jnp.ndarray, optional The frequencies at which to evaluate the PSDs of CARMA process, by default self.frequencies
              plot_lombscargle : bool, optional Plot the Lomb-Scargle periodogram, by default False














      ..
          !! processed by numpydoc !!




Attributes
----------
.. py:data:: __author__
   :value: 'Mehdy Lefkir'

   

.. py:data:: __version__
   :value: '0.1.0'

   



