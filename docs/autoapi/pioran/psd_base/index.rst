
psd_base
========

.. py:module:: pioran.psd_base

.. autoapi-nested-parse::

   Base representation of a power spectral density function.  It is not meant to be used directly, but rather as a base class to build PSDs. 
   The sum and product of PSD are implemented with the ``+`` and ``*`` operators, respectively.

   ..
       !! processed by numpydoc !!


Overview
--------

.. list-table:: Classes
   :header-rows: 0
   :widths: auto
   :class: summarytable

   * - :py:obj:`PowerSpectralDensity <pioran.psd_base.PowerSpectralDensity>`
     - Represents a power density function function, inherited from the :obj:`equinox.Module` class.
   * - :py:obj:`ProductPowerSpectralDensity <pioran.psd_base.ProductPowerSpectralDensity>`
     - Represents the product of two power spectral densities.
   * - :py:obj:`SumPowerSpectralDensity <pioran.psd_base.SumPowerSpectralDensity>`
     - Represents the sum of two power spectral densities.




Classes
-------

.. py:class:: PowerSpectralDensity(param_values: Union[pioran.parameters.ParametersModel, list[float]], param_names: list[str], free_parameters: list[bool])

   Bases: :py:obj:`equinox.Module`

   
   Represents a power density function function, inherited from the :obj:`equinox.Module` class.

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

      * - :py:obj:`parameters <pioran.psd_base.PowerSpectralDensity.parameters>`
        - Parameters of the power spectral density function.
      * - :py:obj:`expression <pioran.psd_base.PowerSpectralDensity.expression>`
        - Expression of the power spectral density function.
      * - :py:obj:`analytical <pioran.psd_base.PowerSpectralDensity.analytical>`
        - If True, the power spectral density function is analytical, otherwise it is not.


   .. list-table:: Methods
      :header-rows: 0
      :widths: auto
      :class: summarytable

      * - :py:obj:`__str__ <pioran.psd_base.PowerSpectralDensity.__str__>`\ ()
        - String representation of the power spectral density.
      * - :py:obj:`__repr__ <pioran.psd_base.PowerSpectralDensity.__repr__>`\ ()
        - Return repr(self).
      * - :py:obj:`__add__ <pioran.psd_base.PowerSpectralDensity.__add__>`\ (other)
        - Overload of the + operator for the power spectral densities.
      * - :py:obj:`__mul__ <pioran.psd_base.PowerSpectralDensity.__mul__>`\ (other)
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



.. py:class:: ProductPowerSpectralDensity(psd1: PowerSpectralDensity, psd2: PowerSpectralDensity)

   Bases: :py:obj:`PowerSpectralDensity`

   
   Represents the product of two power spectral densities.


   :Parameters:

       **psd1** : :obj:`PowerSpectralDensity`
           First power spectral density.

       **psd2** : :obj:`PowerSpectralDensity`
           Second power spectral density.














   ..
       !! processed by numpydoc !!

   .. rubric:: Overview

   .. list-table:: Attributes
      :header-rows: 0
      :widths: auto
      :class: summarytable

      * - :py:obj:`psd1 <pioran.psd_base.ProductPowerSpectralDensity.psd1>`
        - First power spectral density.
      * - :py:obj:`psd2 <pioran.psd_base.ProductPowerSpectralDensity.psd2>`
        - Second power spectral density.
      * - :py:obj:`parameters <pioran.psd_base.ProductPowerSpectralDensity.parameters>`
        - Parameters of the power spectral density.
      * - :py:obj:`expression <pioran.psd_base.ProductPowerSpectralDensity.expression>`
        - Expression of the total power spectral density.


   .. list-table:: Methods
      :header-rows: 0
      :widths: auto
      :class: summarytable

      * - :py:obj:`calculate <pioran.psd_base.ProductPowerSpectralDensity.calculate>`\ (x)
        - Compute the power spectral density at the points x.


   .. rubric:: Members

   .. py:attribute:: psd1
      :type: PowerSpectralDensity

      
      First power spectral density.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: psd2
      :type: PowerSpectralDensity

      
      Second power spectral density.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: parameters
      :type: pioran.parameters.ParametersModel

      
      Parameters of the power spectral density.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: expression
      :type: str

      
      Expression of the total power spectral density.
















      ..
          !! processed by numpydoc !!

   .. py:method:: calculate(x: jax.Array) -> jax.Array

      
      Compute the power spectral density at the points x.

      It is the product of the two power spectral densities.

      :Parameters:

          **x** : :obj:`jax.Array`
              Points where the power spectral density is computed.

      :Returns:

          Product of the two power spectral densitys at the points x.
              ..













      ..
          !! processed by numpydoc !!



.. py:class:: SumPowerSpectralDensity(psd1, psd2)

   Bases: :py:obj:`PowerSpectralDensity`

   
   Represents the sum of two power spectral densities.


   :Parameters:

       **psd1** : :obj:`PowerSpectralDensity`
           First power spectral density.

       **psd2** : :obj:`PowerSpectralDensity`
           Second power spectral density.














   ..
       !! processed by numpydoc !!

   .. rubric:: Overview

   .. list-table:: Attributes
      :header-rows: 0
      :widths: auto
      :class: summarytable

      * - :py:obj:`psd1 <pioran.psd_base.SumPowerSpectralDensity.psd1>`
        - First power spectral density.
      * - :py:obj:`psd2 <pioran.psd_base.SumPowerSpectralDensity.psd2>`
        - Second power spectral density.
      * - :py:obj:`parameters <pioran.psd_base.SumPowerSpectralDensity.parameters>`
        - Parameters of the power spectral density.
      * - :py:obj:`expression <pioran.psd_base.SumPowerSpectralDensity.expression>`
        - Expression of the total power spectral density.


   .. list-table:: Methods
      :header-rows: 0
      :widths: auto
      :class: summarytable

      * - :py:obj:`calculate <pioran.psd_base.SumPowerSpectralDensity.calculate>`\ (x)
        - Compute the power spectrum at the points x.


   .. rubric:: Members

   .. py:attribute:: psd1
      :type: PowerSpectralDensity

      
      First power spectral density.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: psd2
      :type: PowerSpectralDensity

      
      Second power spectral density.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: parameters
      :type: pioran.parameters.ParametersModel

      
      Parameters of the power spectral density.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: expression
      :type: str

      
      Expression of the total power spectral density.
















      ..
          !! processed by numpydoc !!

   .. py:method:: calculate(x: jax.Array) -> jax.Array

      
      Compute the power spectrum at the points x.

      It is the sum of the two power spectra.

      :Parameters:

          **x** : :obj:`jax.Array`
              Points where the power spectrum is computed.

      :Returns:

          :obj:`jax.Array`
              Sum of the two power spectra at the points x.













      ..
          !! processed by numpydoc !!






