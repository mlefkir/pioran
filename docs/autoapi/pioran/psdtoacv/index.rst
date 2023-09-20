
psdtoacv
========

.. py:module:: pioran.psdtoacv

.. autoapi-nested-parse::

   Convert a power spectral density to an autocovariance function via the inverse Fourier transform methods or kernel decomposition.

   ..
       !! processed by numpydoc !!


Overview
--------

.. list-table:: Classes
   :header-rows: 0
   :widths: auto
   :class: summarytable

   * - :py:obj:`PSDToACV <pioran.psdtoacv.PSDToACV>`
     - Represents the tranformation of a power spectral density to an autocovariance function.



.. list-table:: Attributes
   :header-rows: 0
   :widths: auto
   :class: summarytable

   * - :py:obj:`nufft2 <pioran.psdtoacv.nufft2>`
     - \-


Classes
-------

.. py:class:: PSDToACV(PSD: pioran.psd_base.PowerSpectralDensity, S_low: float, S_high: float, T: float, dt: float, method: str, n_components: int = 0, estimate_variance: bool = True, init_variance: float = 1.0)

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

      * - :py:obj:`PSD <pioran.psdtoacv.PSDToACV.PSD>`
        - Power spectral density object.
      * - :py:obj:`ACVF <pioran.psdtoacv.PSDToACV.ACVF>`
        - Autocovariance function as sum of SHO kernels.
      * - :py:obj:`parameters <pioran.psdtoacv.PSDToACV.parameters>`
        - Parameters of the power spectral density.
      * - :py:obj:`method <pioran.psdtoacv.PSDToACV.method>`
        - Method to compute the covariance function from the power spectral density, by default 'FFT'.Possible values are:
      * - :py:obj:`f_max_obs <pioran.psdtoacv.PSDToACV.f_max_obs>`
        - Maximum observed frequency, i.e. the Nyquist frequency.
      * - :py:obj:`f_min_obs <pioran.psdtoacv.PSDToACV.f_min_obs>`
        - Minimum observed frequency.
      * - :py:obj:`f0 <pioran.psdtoacv.PSDToACV.f0>`
        - Lower bound of the frequency grid.
      * - :py:obj:`S_low <pioran.psdtoacv.PSDToACV.S_low>`
        - Scale for the lower bound of the frequency grid.
      * - :py:obj:`S_high <pioran.psdtoacv.PSDToACV.S_high>`
        - Scale for the upper bound of the frequency grid.
      * - :py:obj:`fN <pioran.psdtoacv.PSDToACV.fN>`
        - Upper bound of the frequency grid.
      * - :py:obj:`estimate_variance <pioran.psdtoacv.PSDToACV.estimate_variance>`
        - If True, the amplitude of the autocovariance function is estimated.
      * - :py:obj:`n_freq_grid <pioran.psdtoacv.PSDToACV.n_freq_grid>`
        - Number of points in the frequency grid.
      * - :py:obj:`frequencies <pioran.psdtoacv.PSDToACV.frequencies>`
        - Frequency grid.
      * - :py:obj:`tau <pioran.psdtoacv.PSDToACV.tau>`
        - Time lag grid.
      * - :py:obj:`dtau <pioran.psdtoacv.PSDToACV.dtau>`
        - Time lag step.
      * - :py:obj:`n_components <pioran.psdtoacv.PSDToACV.n_components>`
        - Number of components used to approximate the power spectral density using the 'SHO' method.
      * - :py:obj:`spectral_points <pioran.psdtoacv.PSDToACV.spectral_points>`
        - Frequencies of the SHO kernels.
      * - :py:obj:`spectral_matrix <pioran.psdtoacv.PSDToACV.spectral_matrix>`
        - Matrix of the SHO kernels.


   .. list-table:: Methods
      :header-rows: 0
      :widths: auto
      :class: summarytable

      * - :py:obj:`decompose_model <pioran.psdtoacv.PSDToACV.decompose_model>`\ (psd_normalised)
        - Decompose the model into a sum of SHO kernels.
      * - :py:obj:`get_SHO_coefs <pioran.psdtoacv.PSDToACV.get_SHO_coefs>`\ ()
        - Get the amplitudes and frequencies of the SHO kernels.
      * - :py:obj:`build_SHO_model <pioran.psdtoacv.PSDToACV.build_SHO_model>`\ (amplitudes, frequencies)
        - Build the semi-separable SHO model in tinygp from the amplitudes and frequencies.
      * - :py:obj:`calculate_rescale <pioran.psdtoacv.PSDToACV.calculate_rescale>`\ (t)
        - \-
      * - :py:obj:`calculate <pioran.psdtoacv.PSDToACV.calculate>`\ (t, with_ACVF_factor)
        - Calculate the autocovariance function from the power spectral density.
      * - :py:obj:`get_acvf_byNuFFT <pioran.psdtoacv.PSDToACV.get_acvf_byNuFFT>`\ (psd, t)
        - Compute the autocovariance function from the power spectral density using the non uniform Fourier transform.
      * - :py:obj:`get_acvf_byFFT <pioran.psdtoacv.PSDToACV.get_acvf_byFFT>`\ (psd)
        - Compute the autocovariance function from the power spectral density using the inverse Fourier transform.
      * - :py:obj:`interpolation <pioran.psdtoacv.PSDToACV.interpolation>`\ (t, acvf)
        - Interpolate the autocovariance function at the points t.
      * - :py:obj:`get_cov_matrix <pioran.psdtoacv.PSDToACV.get_cov_matrix>`\ (xq, xp)
        - Compute the covariance matrix between two arrays xq, xp.
      * - :py:obj:`__str__ <pioran.psdtoacv.PSDToACV.__str__>`\ ()
        - String representation of the PSDToACV object.
      * - :py:obj:`__repr__ <pioran.psdtoacv.PSDToACV.__repr__>`\ ()
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
      :type: Union[int, None]

      
      Number of points in the frequency grid.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: frequencies
      :type: Union[jax.Array, None]

      
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
      :type: Union[jax.Array, None]

      
      Frequencies of the SHO kernels.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: spectral_matrix
      :type: Union[jax.Array, None]

      
      Matrix of the SHO kernels.
















      ..
          !! processed by numpydoc !!

   .. py:method:: decompose_model(psd_normalised: jax.Array)

      
      Decompose the model into a sum of SHO kernels.

      Assuming that the model can be written as a sum of :math:`J` SHO kernels, this method
      solve the linear system to find the amplitude :math:`a_j` of each kernel.

      .. math:: :label: sho_power_spectrum

      \boldsymbol{y} = B \boldsymbol{a}

      with :math:`\boldsymbol{y}=\begin{bmatrix}1 & \mathcal{P}(f_1)/\mathcal{P}(f_0) & \cdots & \mathcal{P}(f_J)/\mathcal{P}(f_0) \end{bmatrix}^\mathrm{T}`
      the normalised power spectral density vector, :math:`B` the spectral matrix associated to the linear system and :math:`\boldsymbol{a}` the amplitudes of the SHO kernels.

      .. math:: :label: sho_spectral_matrix

      B_{ij} = \dfrac{1}{1 + \left(\dfrac{f_i}{f_j}\right)^4}

      :Parameters:

          **psd_normalised** : :obj:`jax.Array`
              Normalised power spectral density by the first value.

      :Returns:

          :obj:`jax.Array`
              Amplitudes of the SHO kernels.

          :obj:`jax.Array`
              Frequencies of the SHO kernels.













      ..
          !! processed by numpydoc !!

   .. py:method:: get_SHO_coefs()

      
      Get the amplitudes and frequencies of the SHO kernels.

      Estimate the amplitudes and frequencies of the SHO kernels by solving the linear system.


      :Returns:

          **amplitudes** : :obj:`jax.Array`
              Amplitudes of the SHO kernels.

          **frequencies** : :obj:`jax.Array`
              Frequencies of the SHO kernels.













      ..
          !! processed by numpydoc !!

   .. py:method:: build_SHO_model(amplitudes: jax.Array, frequencies: jax.Array) -> tinygp.kernels.quasisep.SHO

      
      Build the semi-separable SHO model in tinygp from the amplitudes and frequencies.


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

   .. py:method:: calculate_rescale(t: jax.Array) -> jax.Array


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




Attributes
----------
.. py:data:: nufft2

   



