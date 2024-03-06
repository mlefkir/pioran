
simulate
========

.. py:module:: pioran.simulate

.. autoapi-nested-parse::

   Generate time series from a given PSD or ACVF model.

   ..
       !! processed by numpydoc !!


Overview
--------

.. list-table:: Classes
   :header-rows: 0
   :widths: auto
   :class: summarytable

   * - :py:obj:`Simulations <pioran.simulate.Simulations>`
     - Simulate time series from a given PSD or ACVF model.




Classes
-------

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

      * - :py:obj:`duration <pioran.simulate.Simulations.duration>`
        - duration of the time series.
      * - :py:obj:`sampling_period <pioran.simulate.Simulations.sampling_period>`
        - sampling period of the time series.
      * - :py:obj:`n_time <pioran.simulate.Simulations.n_time>`
        - number of time indexes.
      * - :py:obj:`t <pioran.simulate.Simulations.t>`
        - time :obj:`jax.Array` of the time series.
      * - :py:obj:`f_max_obs <pioran.simulate.Simulations.f_max_obs>`
        - maximum frequency of the observed frequency grid.
      * - :py:obj:`f_min_obs <pioran.simulate.Simulations.f_min_obs>`
        - minimum frequency of the observed frequency grid.
      * - :py:obj:`f0 <pioran.simulate.Simulations.f0>`
        - minimum frequency of the total frequency grid.
      * - :py:obj:`fN <pioran.simulate.Simulations.fN>`
        - maximum frequency of the total frequency grid.
      * - :py:obj:`n_freq_grid <pioran.simulate.Simulations.n_freq_grid>`
        - number of frequency indexes.
      * - :py:obj:`frequencies <pioran.simulate.Simulations.frequencies>`
        - frequency array of the total frequency grid.
      * - :py:obj:`tau_max <pioran.simulate.Simulations.tau_max>`
        - maximum lag of the autocovariance function.
      * - :py:obj:`dtau <pioran.simulate.Simulations.dtau>`
        - sampling period of the autocovariance function.
      * - :py:obj:`tau <pioran.simulate.Simulations.tau>`
        - lag array of the autocovariance function.
      * - :py:obj:`psd <pioran.simulate.Simulations.psd>`
        - power spectral density of the time series.
      * - :py:obj:`acvf <pioran.simulate.Simulations.acvf>`
        - autocovariance function of the time series.
      * - :py:obj:`triang <pioran.simulate.Simulations.triang>`
        - triangular matrix used to generate the time series with the Cholesky decomposition.
      * - :py:obj:`keys <pioran.simulate.Simulations.keys>`
        - dictionary of the keys used to generate the random numbers. See :func:`~pioran.simulate.Simulations.generate_keys` for more details.


   .. list-table:: Methods
      :header-rows: 0
      :widths: auto
      :class: summarytable

      * - :py:obj:`generate_keys <pioran.simulate.Simulations.generate_keys>`\ (seed)
        - Generate the keys to generate the random numbers.
      * - :py:obj:`plot_acvf <pioran.simulate.Simulations.plot_acvf>`\ (figsize, xunit, filename, title)
        - Plot the autocovariance function.
      * - :py:obj:`plot_psd <pioran.simulate.Simulations.plot_psd>`\ (figsize, filename, title, xunit, loglog)
        - Plot the power spectral density model.
      * - :py:obj:`GP_method <pioran.simulate.Simulations.GP_method>`\ (t_test, interpolation)
        - Generate a time series using the GP method.
      * - :py:obj:`simulate <pioran.simulate.Simulations.simulate>`\ (mean, method, irregular_sampling, irregular_gaps, randomise_fluxes, errors, errors_size, N_points, seed, seed_gaps, filename, exponentiate_ts, min_n_gaps, max_n_gaps, max_size_slices, interp_method)
        - Method to simulate time series using either the GP method or the TK method.
      * - :py:obj:`simulate_irregular_gaps <pioran.simulate.Simulations.simulate_irregular_gaps>`\ (a, seed, N_points, min_n_gaps, max_n_gaps, max_size_slices)
        - Simulate irregular times from a regular time series with random gaps.
      * - :py:obj:`extract_subset_timeseries <pioran.simulate.Simulations.extract_subset_timeseries>`\ (t, y, M)
        - Select a random subset of points from an input time series.
      * - :py:obj:`sample_timeseries <pioran.simulate.Simulations.sample_timeseries>`\ (t, y, M, irregular_sampling)
        - Extract a random subset of points from the time series.
      * - :py:obj:`timmer_Koenig_method <pioran.simulate.Simulations.timmer_Koenig_method>`\ ()
        - Generate a time series using the Timmer-Konig method.
      * - :py:obj:`split_longtimeseries <pioran.simulate.Simulations.split_longtimeseries>`\ (t, ts, n_slices)
        - Split a long time series into shorter time series.
      * - :py:obj:`resample_longtimeseries <pioran.simulate.Simulations.resample_longtimeseries>`\ (t_slices, ts_slices)
        - Resample the time series to have a regular sampling period with n_time points.
      * - :py:obj:`simulate_longtimeseries <pioran.simulate.Simulations.simulate_longtimeseries>`\ (mean, randomise_fluxes, errors, seed)
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






