
diagnostics
===========

.. py:module:: pioran.diagnostics


Overview
--------

.. list-table:: Classes
   :header-rows: 0
   :widths: auto
   :class: summarytable

   * - :py:obj:`Visualisations <pioran.diagnostics.Visualisations>`
     - Class for visualising the results after an inference run.




Classes
-------

.. py:class:: Visualisations(process: Union[pioran.core.GaussianProcess, pioran.carma.carma_core.CARMAProcess], filename: str, n_frequencies: int = 2500)

   
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

      * - :py:obj:`process <pioran.diagnostics.Visualisations.process>`
        - The process to be visualised.
      * - :py:obj:`x <pioran.diagnostics.Visualisations.x>`
        - The observation times.
      * - :py:obj:`y <pioran.diagnostics.Visualisations.y>`
        - The observation values.
      * - :py:obj:`yerr <pioran.diagnostics.Visualisations.yerr>`
        - The observation errors.
      * - :py:obj:`predictive_mean <pioran.diagnostics.Visualisations.predictive_mean>`
        - The predictive mean.
      * - :py:obj:`predictive_cov <pioran.diagnostics.Visualisations.predictive_cov>`
        - The predictive covariance.
      * - :py:obj:`x_pred <pioran.diagnostics.Visualisations.x_pred>`
        - The prediction times.
      * - :py:obj:`f_min <pioran.diagnostics.Visualisations.f_min>`
        - The minimum frequency.
      * - :py:obj:`f_max <pioran.diagnostics.Visualisations.f_max>`
        - The maximum frequency.
      * - :py:obj:`frequencies <pioran.diagnostics.Visualisations.frequencies>`
        - The frequencies at which to evaluate the PSDs.
      * - :py:obj:`tau <pioran.diagnostics.Visualisations.tau>`
        - The times at which to evaluate the ACFs.
      * - :py:obj:`filename_prefix <pioran.diagnostics.Visualisations.filename_prefix>`
        - The filename prefix for the output plots.


   .. list-table:: Methods
      :header-rows: 0
      :widths: auto
      :class: summarytable

      * - :py:obj:`plot_timeseries_diagnostics <pioran.diagnostics.Visualisations.plot_timeseries_diagnostics>`\ (prediction_indexes, \*\*kwargs)
        - Plot the timeseries diagnostics.
      * - :py:obj:`posterior_predictive_checks <pioran.diagnostics.Visualisations.posterior_predictive_checks>`\ (samples, plot_PSD, plot_ACVF, \*\*kwargs)
        - Plot the posterior predictive checks.


   .. rubric:: Members

   .. py:attribute:: process
      :type: Union[pioran.core.GaussianProcess, pioran.carma.carma_core.CARMAProcess]

      
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

   .. py:method:: plot_timeseries_diagnostics(prediction_indexes: Union[jax.Array, None] = None, **kwargs) -> None

      
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
              frequencies : jnp.ndarray, optional
                  The frequencies at which to evaluate the PSDs of CARMA process, by default self.frequencies
              plot_lombscargle : bool, optional
                  Plot the Lomb-Scargle periodogram, by default False














      ..
          !! processed by numpydoc !!






