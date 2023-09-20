
psd
===

.. py:module:: pioran.psd


Overview
--------

.. list-table:: Classes
   :header-rows: 0
   :widths: auto
   :class: summarytable

   * - :py:obj:`Lorentzian <pioran.psd.Lorentzian>`
     - Lorentzian power spectral density.
   * - :py:obj:`Gaussian <pioran.psd.Gaussian>`
     - Gaussian power spectral density.
   * - :py:obj:`OneBendPowerLaw <pioran.psd.OneBendPowerLaw>`
     - One-bend power-law power spectral density.
   * - :py:obj:`Matern32PSD <pioran.psd.Matern32PSD>`
     - Power spectral density of the Matern 3/2 covariance function.




Classes
-------

.. py:class:: Lorentzian(parameters_values: list, free_parameters: list = [True, True, True])

   Bases: :py:obj:`pioran.psd_base.PowerSpectralDensity`

   
   Lorentzian power spectral density.

   .. math:: :label: lorentzianpsd 

      \mathcal{P}(f) = \dfrac{A}{\gamma^2 +4\pi^2 (f-f_0)^2}.

   with the amplitude :math:`A\ge 0`, the position :math:`f_0\ge 0` and the halfwidth :math:`\gamma>0`.

   The parameters are stored in the `parameters` attribute which is a :class:`~pioran.parameters.ParametersModel` object. 
   The values of the parameters can be accessed using the `parameters` attribute via three keys: '`position`', '`amplitude`' and '`halfwidth`'.

   The power spectral density function is evaluated on an array of frequencies :math:`f` using the `calculate` method.

   :Parameters:

       **param_values** : :obj:`list` of :obj:`float`
           Values of the parameters of the power spectral density function. [position, amplitude, halfwidth]

       **free_parameters: :obj:`list` of :obj:`bool`, optional**
           List of bool to indicate if the parameters are free or not. Default is `[True, True,True]`.














   ..
       !! processed by numpydoc !!

   .. rubric:: Overview

   .. list-table:: Attributes
      :header-rows: 0
      :widths: auto
      :class: summarytable

      * - :py:obj:`parameters <pioran.psd.Lorentzian.parameters>`
        - Parameters of the power spectral density function.
      * - :py:obj:`expression <pioran.psd.Lorentzian.expression>`
        - Expression of the power spectral density function.
      * - :py:obj:`analytical <pioran.psd.Lorentzian.analytical>`
        - If True, the power spectral density function is analytical, otherwise it is not.


   .. list-table:: Methods
      :header-rows: 0
      :widths: auto
      :class: summarytable

      * - :py:obj:`calculate <pioran.psd.Lorentzian.calculate>`\ (f)
        - Computes the power spectral density.


   .. rubric:: Members

   .. py:attribute:: parameters
      :type: pioran.parameters.ParametersModel

      
      Parameters of the power spectral density function.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: expression
      :value: 'lorentzian'

      
      Expression of the power spectral density function.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: analytical
      :value: True

      
      If True, the power spectral density function is analytical, otherwise it is not.
















      ..
          !! processed by numpydoc !!

   .. py:method:: calculate(f) -> jax.Array

      
      Computes the power spectral density.        

      The expression is given by Equation :math:numref:`lorentzianpsd`.
      with the variance :math:`A\ge 0`, the position :math:`f_0\ge 0` and the halfwidth :math:`\gamma>0`.

      :Parameters:

          **f** : :obj:`jax.Array`
              Array of frequencies.

      :Returns:

          :obj:`jax.Array`
              Power spectral density function evaluated on the array of frequencies.













      ..
          !! processed by numpydoc !!



.. py:class:: Gaussian(parameters_values, free_parameters=[True, True, True])

   Bases: :py:obj:`pioran.psd_base.PowerSpectralDensity`

   
   Gaussian power spectral density.

   .. math:: :label: gaussianpsd 

      \mathcal{P}(f) = \dfrac{A}{\sqrt{2\pi}\sigma} \exp\left(-\dfrac{\left(f-f_0\right)^2}{2\sigma^2} \right).

   with the amplitude :math:`A\ge 0`, the position :math:`f_0\ge 0` and the standard-deviation '`sigma`' :math:`\sigma>0`.

   The parameters are stored in the `parameters` attribute which is a :class:`~pioran.parameters.ParametersModel` object. 
   The values of the parameters can be accessed using the `parameters` attribute via three keys: '`position`', '`amplitude`' and '`sigma`'

   The power spectral density function is evaluated on an array of frequencies :math:`f` using the `calculate` method.

   :Parameters:

       **param_values** : :obj:`list` of :obj:`float`
           Values of the parameters of the power spectral density function.

       **free_parameters** : :obj:`list` of :obj:`bool`, optional
           List of bool to indicate if the parameters are free or not. Default is `[True, True,True]`.














   ..
       !! processed by numpydoc !!

   .. rubric:: Overview

   .. list-table:: Attributes
      :header-rows: 0
      :widths: auto
      :class: summarytable

      * - :py:obj:`expression <pioran.psd.Gaussian.expression>`
        - Expression of the power spectral density function.
      * - :py:obj:`parameters <pioran.psd.Gaussian.parameters>`
        - Expression of the power spectral density function.
      * - :py:obj:`analytical <pioran.psd.Gaussian.analytical>`
        - If True, the power spectral density function is analytical, otherwise it is not.


   .. list-table:: Methods
      :header-rows: 0
      :widths: auto
      :class: summarytable

      * - :py:obj:`calculate <pioran.psd.Gaussian.calculate>`\ (f)
        - Computes the power spectral density.


   .. rubric:: Members

   .. py:attribute:: expression
      :value: 'gaussian'

      
      Expression of the power spectral density function.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: parameters
      :type: pioran.parameters.ParametersModel

      
      Expression of the power spectral density function.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: analytical
      :value: True

      
      If True, the power spectral density function is analytical, otherwise it is not.
















      ..
          !! processed by numpydoc !!

   .. py:method:: calculate(f) -> jax.Array

      
      Computes the power spectral density.

      The expression is given by Equation :math:numref:`gaussianpsd` 
      with the variance :math:`A\ge 0`, the position :math:`f_0\ge 0` and the standard-deviation :math:`\sigma>0`.

      :Parameters:

          **f** : :obj:`jax.Array`
              Array of frequencies.

      :Returns:

          :obj:`jax.Array`
              Power spectral density function evaluated on the array of frequencies.













      ..
          !! processed by numpydoc !!



.. py:class:: OneBendPowerLaw(parameters_values, free_parameters=[False, True, True, True])

   Bases: :py:obj:`pioran.psd_base.PowerSpectralDensity`

   
   One-bend power-law power spectral density.

   .. math:: :label: onebendpowerlawpsd

       \mathcal{P}(f) = A\times (f/f_1)^{\alpha_1} \frac{1}{1+(f/f_1)^{(\alpha_1-\alpha_2)}}.

   with the amplitude :math:`A\ge 0`, the bend frequency :math:`f_1\ge 0` and the indices :math:`\alpha_1,\alpha_2`.

   :Parameters:

       **param_values** : :obj:`list` of :obj:`float`
           Values of the parameters of the power spectral density function. 
           In order: [norm, index_1, freq_1, index_2]

       **free_parameters** : :obj:`list` of :obj:`bool`, optional
           List of bool to indicate if the parameters are free or not. Default is `[False, True, True,True]`.














   ..
       !! processed by numpydoc !!

   .. rubric:: Overview

   .. list-table:: Attributes
      :header-rows: 0
      :widths: auto
      :class: summarytable

      * - :py:obj:`expression <pioran.psd.OneBendPowerLaw.expression>`
        - Expression of the power spectral density function.
      * - :py:obj:`parameters <pioran.psd.OneBendPowerLaw.parameters>`
        - Parameters of the power spectral density function.
      * - :py:obj:`analytical <pioran.psd.OneBendPowerLaw.analytical>`
        - If True, the power spectral density function is analytical, otherwise it is not.


   .. list-table:: Methods
      :header-rows: 0
      :widths: auto
      :class: summarytable

      * - :py:obj:`calculate <pioran.psd.OneBendPowerLaw.calculate>`\ (f)
        - Computes the power spectral density.


   .. rubric:: Members

   .. py:attribute:: expression
      :value: 'onebend-powerlaw'

      
      Expression of the power spectral density function.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: parameters
      :type: pioran.parameters.ParametersModel

      
      Parameters of the power spectral density function.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: analytical
      :value: False

      
      If True, the power spectral density function is analytical, otherwise it is not.
















      ..
          !! processed by numpydoc !!

   .. py:method:: calculate(f)

      
      Computes the power spectral density.        

      The expression is given by Equation :math:numref:`onebendpowerlawpsd`
      with the variance :math:`A\ge 0` and the scale :math:`\gamma>0`.

      :Parameters:

          **f** : :obj:`jax.Array`
              Array of frequencies.

      :Returns:

          :obj:`jax.Array`
              Power spectral density function evaluated on the array of frequencies.













      ..
          !! processed by numpydoc !!



.. py:class:: Matern32PSD(parameters_values, free_parameters=[True, True])

   Bases: :py:obj:`pioran.psd_base.PowerSpectralDensity`

   
   Power spectral density of the Matern 3/2 covariance function.

   .. math:: :label: matern32psd 

      \mathcal{P}(f) = \dfrac{A}{\gamma^3}\dfrac{12\sqrt{3}}{{(3/\gamma^2 +4\pi^2 f^2)}^2}.

   with the amplitude :math:`A\ge 0` and the scale :math:`\gamma>0`.

   The parameters are stored in the `parameters` attribute which is a :class:`~pioran.parameters.ParametersModel` object. 
   The values of the parameters can be accessed using the `parameters` attribute via three keys: '`position`' and '`scale`'

   The power spectral density function is evaluated on an array of frequencies :math:`f` using the `calculate` method.

   :Parameters:

       **param_values** : :obj:`list of float`
           Values of the parameters of the power spectral density function.

       **free_parameters** : :obj:`list` of :obj:`bool`, optional
           List of bool to indicate if the parameters are free or not. Default is `[True,True]`.














   ..
       !! processed by numpydoc !!

   .. rubric:: Overview

   .. list-table:: Attributes
      :header-rows: 0
      :widths: auto
      :class: summarytable

      * - :py:obj:`parameters <pioran.psd.Matern32PSD.parameters>`
        - Parameters of the power spectral density function.
      * - :py:obj:`expression <pioran.psd.Matern32PSD.expression>`
        - Expression of the power spectral density function.
      * - :py:obj:`analytical <pioran.psd.Matern32PSD.analytical>`
        - If True, the power spectral density function is analytical, otherwise it is not.


   .. list-table:: Methods
      :header-rows: 0
      :widths: auto
      :class: summarytable

      * - :py:obj:`calculate <pioran.psd.Matern32PSD.calculate>`\ (f)
        - Computes the power spectral density.


   .. rubric:: Members

   .. py:attribute:: parameters
      :type: pioran.parameters.ParametersModel

      
      Parameters of the power spectral density function.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: expression
      :value: 'matern32psd'

      
      Expression of the power spectral density function.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: analytical
      :value: True

      
      If True, the power spectral density function is analytical, otherwise it is not.
















      ..
          !! processed by numpydoc !!

   .. py:method:: calculate(f) -> jax.Array

      
      Computes the power spectral density.

      The expression is given by Equation :math:numref:`matern32psd`
      with the variance :math:`A\ge 0` and the scale :math:`\gamma>0`.

      :Parameters:

          **f** : :obj:`jax.Array`
              Array of frequencies.

      :Returns:

          :obj:`jax.Array`
              Power spectral density function evaluated on the array of frequencies.













      ..
          !! processed by numpydoc !!






