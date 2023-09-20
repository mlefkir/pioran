
plots
=====

.. py:module:: pioran.plots

.. autoapi-nested-parse::

   Collection of functions for plotting the results of the Gaussian Process Regression.

   ..
       !! processed by numpydoc !!


Overview
--------


.. list-table:: Function
   :header-rows: 0
   :widths: auto
   :class: summarytable

   * - :py:obj:`plot_prediction <pioran.plots.plot_prediction>`\ (x, y, yerr, x_pred, y_pred, cov_pred, filename, log_transform, figsize, confidence_bands, title, xlabel, ylabel, xlim, ylim, \*\*kwargs)
     - Plot the predicted time series and the confidence bands.
   * - :py:obj:`plot_residuals <pioran.plots.plot_residuals>`\ (x, y, yerr, y_pred, filename, log_transform, confidence_intervals, figsize, maxlag, title, \*\*kwargs)
     - Plot the residuals of a predicted time series.
   * - :py:obj:`plot_posterior_predictive_ACF <pioran.plots.plot_posterior_predictive_ACF>`\ (tau, acf, x, y, filename, with_mean, confidence_bands, xlabel, save_data, \*\*kwargs)
     - Plot the posterior predictive Autocorrelation function of the process.
   * - :py:obj:`plot_posterior_predictive_PSD <pioran.plots.plot_posterior_predictive_PSD>`\ (f, posterior_PSD, x, y, yerr, filename, posterior_PSD_approx, plot_lombscargle, save_data, with_mean, confidence_bands, ylim, xlabel, f_min_obs, f_max_obs, \*\*kwargs)
     - Plot the posterior predictive Power Spectral Density of the process.
   * - :py:obj:`diagnostics_psd_approx <pioran.plots.diagnostics_psd_approx>`\ (f, res, ratio, f_min, f_max)
     - Plot the mean residuals and the ratios of the PSD approximation as a function of frequency
   * - :py:obj:`violin_plots_psd_approx <pioran.plots.violin_plots_psd_approx>`\ (res, ratio)
     - Plot the violin plots of the residuals and the ratios of the PSD approximation.
   * - :py:obj:`residuals_quantiles <pioran.plots.residuals_quantiles>`\ (residuals, ratio, f, f_min, f_max)
     - Plot the quantiles of the residuals and the ratios of the PSD approximation as a function of frequency.
   * - :py:obj:`plot_priors_samples <pioran.plots.plot_priors_samples>`\ (params, names)
     - Plot the samples from the prior distributions.
   * - :py:obj:`plot_prior_predictive_PSD <pioran.plots.plot_prior_predictive_PSD>`\ (f, psd_samples, xlim, ylim, xunit)
     - Plot the prior predictive Power Spectral Density of the process.




Functions
---------
.. py:function:: plot_prediction(x, y, yerr, x_pred, y_pred, cov_pred, filename, log_transform=False, figsize=(16, 6), confidence_bands=True, title=None, xlabel='Time', ylabel=None, xlim=None, ylim=None, **kwargs)

   
   Plot the predicted time series and the confidence bands.


   :Parameters:

       **x** : :obj:`numpy.ndarray`
           Time of the observations.

       **y** : :obj:`numpy.ndarray`
           Values of the observations.

       **yerr** : :obj:`numpy.ndarray`
           Error of the observations.

       **x_pred** : :obj:`numpy.ndarray`
           Time of the prediction.

       **y_pred** : :obj:`numpy.ndarray`
           Values of the prediction.

       **cov_pred** : :obj:`numpy.ndarray`
           Covariance matrix of the prediction.    

       **filename** : :obj:`str`
           Name of the file to save the figure.

       **log_transform** : :obj:`bool`, optional
           Log transform the prediction, by default False

       **figsize** : :obj:`tuple`, optional
           Size of the figure.

       **confidence_bands** : :obj:`bool`, optional
           Plot the confidence bands, by default True

       **title** : :obj:`str`, optional
           Title of the plot, by default None

       **xlabel** : :obj:`str`, optional
           Label for the x-axis, by default None

       **ylabel** : :obj:`str`, optional
           Label for the y-axis, by default None

       **xlim** : :obj:`tuple` of :obj:`float`, optional
           Limits of the x-axis, by default None

       **ylim** : :obj:`tuple` of :obj:`float`, optional
           Limits of the y-axis, by default None

   :Returns:

       **fig** : :obj:`matplotlib.figure.Figure`
           Figure object.

       **ax** : :obj:`matplotlib.axes.Axes`
           Axes object.













   ..
       !! processed by numpydoc !!

.. py:function:: plot_residuals(x, y, yerr, y_pred, filename, log_transform=False, confidence_intervals=[95, 99], figsize=(10, 10), maxlag=None, title=None, **kwargs)

   
   Plot the residuals of a predicted time series.


   :Parameters:

       **x** : :obj:`numpy.ndarray`
           Time of the observations.

       **y** : :obj:`numpy.ndarray`
           Values of the observations.

       **yerr** : :obj:`numpy.ndarray`
           Error of the observations.

       **y_pred** : :obj:`numpy.ndarray`
           Values of the prediction at the time of the observations.

       **filename** : :obj:`str`
           Name of the file to save the figure.

       **log_transform** : :obj:`bool`, optional
           Log transform the prediction, by default False

       **confidence_intervals** : :obj:`list` of :obj:`float`, optional
           Confidence intervals to plot, by default [95,99]

       **figsize** : :obj:`tuple`, optional
           Size of the figure.

       **maxlag** : :obj:`int`, optional
           Maximum lag to plot, by default None

       **title** : :obj:`str`, optional
           Title of the plot, by default None

   :Returns:

       **fig** : :obj:`matplotlib.figure.Figure`
           Figure object.

       **ax** : :obj:`matplotlib.axes.Axes`
           Axes object.













   ..
       !! processed by numpydoc !!

.. py:function:: plot_posterior_predictive_ACF(tau, acf, x, y, filename, with_mean=False, confidence_bands=[68, 95], xlabel='Time lag (d)', save_data=False, **kwargs)

   
   Plot the posterior predictive Autocorrelation function of the process.

   This function will also compute the interpolated cross-correlation function using the 
   code from https://bitbucket.org/cgrier/python_ccf_code/src/master/   

   :Parameters:

       **tau** : :obj:`numpy.ndarray`
           Time lags.

       **acf** : :obj:`numpy.ndarray`
           Array of ACFs posterior samples.

       **x** : :obj:`numpy.ndarray`
           Time indexes.

       **y** : :obj:`numpy.ndarray`
           Time series values.

       **filename** : :obj:`str`
           Filename to save the figure.

       **with_mean** : bool, optional
           Plot the mean of the samples, by default False

       **confidence_bands** : list, optional
           Confidence intervals to plot, by default [95,99]

       **xlabel** : str, optional
           , by default r'Time lag (d)'

       **save_data** : bool, optional
           Save the data to a text file, by default False

   :Returns:

       **fig** : :obj:`matplotlib.figure.Figure`
           Figure object.

       **ax** : :obj:`matplotlib.axes.Axes`
           Axes object.













   ..
       !! processed by numpydoc !!

.. py:function:: plot_posterior_predictive_PSD(f, posterior_PSD, x, y, yerr, filename, posterior_PSD_approx=None, plot_lombscargle=False, save_data=False, with_mean=False, confidence_bands=[68, 95], ylim=None, xlabel='Frequency $\\mathrm{d}^{-1}$', f_min_obs=None, f_max_obs=None, **kwargs)

   
   Plot the posterior predictive Power Spectral Density of the process.

   This function will also compute the Lomb-Scargle periodogram on the data.

   :Parameters:

       **f** : :obj:`numpy.ndarray`
           Frequencies.

       **posterior_PSD** : :obj:`numpy.ndarray`
           Array of PSDs posterior samples.

       **x** : :obj:`numpy.ndarray`
           Time indexes.

       **y** : :obj:`numpy.ndarray`
           Time series values.

       **yerr** : :obj:`numpy.ndarray`
           Time series errors.  

       **posterior_PSD_approx** : :obj:`numpy.ndarray`, optional
           Array of PSDs posterior samples for the approximation, by default None

       **filename** : :obj:`str`
           Filename to save the figure.

       **with_mean** : bool, optional
           Plot the mean of the samples, by default False

       **confidence_bands** : list, optional
           Confidence intervals to plot, by default [95,99]

       **xlabel** : str, optional
           , by default r'Time lag (d)'

       **save_data** : bool, optional
           Save the data to a text file, by default False

   :Returns:

       **fig** : :obj:`matplotlib.figure.Figure`
           Figure object.

       **ax** : :obj:`matplotlib.axes.Axes`
           Axes object.













   ..
       !! processed by numpydoc !!

.. py:function:: diagnostics_psd_approx(f, res, ratio, f_min, f_max)

   
   Plot the mean residuals and the ratios of the PSD approximation as a function of frequency


   :Parameters:

       **f** : :obj:`jax.Array`
           Frequency array.

       **res** : :obj:`jax.Array`
           Residuals of the PSD approximation.

       **ratio** : :obj:`jax.Array`
           Ratio of the PSD approximation.

       **f_min** : :obj:`float`
           Minimum observed frequency.

       **f_max** : :obj:`float`
           Maximum observed frequency.

   :Returns:

       **fig** : :obj:`matplotlib.figure.Figure`
           Figure object.

       **ax** : :obj:`matplotlib.axes.Axes`
           Axes object.













   ..
       !! processed by numpydoc !!

.. py:function:: violin_plots_psd_approx(res, ratio)

   
   Plot the violin plots of the residuals and the ratios of the PSD approximation.


   :Parameters:

       **res** : :obj:`jax.Array`
           Residuals of the PSD approximation.

       **ratio** : :obj:`jax.Array`
           Ratios of the PSD approximation.

   :Returns:

       **fig** : :obj:`matplotlib.figure.Figure`
           Figure object.

       **ax** : :obj:`matplotlib.axes.Axes`
           Axes object.













   ..
       !! processed by numpydoc !!

.. py:function:: residuals_quantiles(residuals, ratio, f, f_min, f_max)

   
   Plot the quantiles of the residuals and the ratios of the PSD approximation as a function of frequency.


   :Parameters:

       **res** : :obj:`jax.Array`
           Residuals of the PSD approximation.

       **ratio** : :obj:`jax.Array`
           Ratios of the PSD approximation.

       **f** : :obj:`jax.Array`
           Frequency array.

       **f_min** : :obj:`float`
           Minimum observed frequency.

       **f_max** : :obj:`float`
           Maximum observed frequency.

   :Returns:

       **fig** : :obj:`matplotlib.figure.Figure`
           Figure object.

       **ax** : :obj:`matplotlib.axes.Axes`
           Axes object.













   ..
       !! processed by numpydoc !!

.. py:function:: plot_priors_samples(params, names)

   
   Plot the samples from the prior distributions.


   :Parameters:

       **params** : :obj:`numpy.ndarray`
           Array of samples from the prior distributions.

       **names** : :obj:`list` of :obj:`str`
           Names of the parameters.

   :Returns:

       :obj:`matplotlib.figure.Figure`
           Figure object.

       :obj:`matplotlib.axes.Axes`
           Axes object.    













   ..
       !! processed by numpydoc !!

.. py:function:: plot_prior_predictive_PSD(f, psd_samples, xlim=(None, None), ylim=(None, None), xunit='$d^{-1}$')

   
   Plot the prior predictive Power Spectral Density of the process.


   :Parameters:

       **f** : :obj:`numpy.ndarray`
           Frequencies.

       **psd_samples** : :obj:`numpy.ndarray`
           Array of PSDs posterior samples.

       **xlim** : tuple, optional
           Limits on the x-axis, by default (None,None)

       **ylim** : tuple, optional
           Limits on the y-axis, by default (None,None)

       **xunit** : str, optional
           Unit of the xaxis, by default r'$d^{-1}$'














   ..
       !! processed by numpydoc !!




