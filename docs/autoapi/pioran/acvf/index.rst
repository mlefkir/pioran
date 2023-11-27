
acvf
====

.. py:module:: pioran.acvf

.. autoapi-nested-parse::

   Collection of covariance functions.

   ..
       !! processed by numpydoc !!


Overview
--------

.. list-table:: Classes
   :header-rows: 0
   :widths: auto
   :class: summarytable

   * - :py:obj:`Exponential <pioran.acvf.Exponential>`
     - Exponential covariance function.
   * - :py:obj:`SquaredExponential <pioran.acvf.SquaredExponential>`
     - Squared exponential covariance function.
   * - :py:obj:`Matern32 <pioran.acvf.Matern32>`
     - Matern 3/2 covariance function.
   * - :py:obj:`Matern52 <pioran.acvf.Matern52>`
     - Matern 5/2 covariance function.
   * - :py:obj:`RationalQuadratic <pioran.acvf.RationalQuadratic>`
     - Rational quadratic covariance function.




Classes
-------

.. py:class:: Exponential(param_values: list[float], free_parameters: list[bool] = [True, True])

   Bases: :py:obj:`pioran.acvf_base.CovarianceFunction`

   
   Exponential covariance function.

   .. math:: :label: expocov

      K(\tau) = \dfrac{A}{2\gamma} \times \exp( {- |\tau| \gamma}).

   with the variance :math:`A\ge 0` and length :math:`\gamma>0`.

   The parameters are stored in the `parameters` attribute which is a :class:`~pioran.parameters.ParametersModel` object.
   The values of the parameters can be accessed using the `parameters` attribute via two keys: '`variance`' and '`length`'.

   The covariance function is evaluated on an array of lags :math:`\tau` using the `calculate` method.

   :Parameters:

       **param_values** : :obj:`list` of :obj:`float`
           Values of the parameters of the covariance function. [`variance`, `length`]

       **free_parameters** : :obj:`list` of :obj:`bool`
           List of bool to indicate if the parameters are free or not.














   ..
       !! processed by numpydoc !!

   .. rubric:: Overview

   .. list-table:: Attributes
      :header-rows: 0
      :widths: auto
      :class: summarytable

      * - :py:obj:`parameters <pioran.acvf.Exponential.parameters>`
        - Parameters of the covariance function.
      * - :py:obj:`expression <pioran.acvf.Exponential.expression>`
        - Expression of the covariance function.


   .. list-table:: Methods
      :header-rows: 0
      :widths: auto
      :class: summarytable

      * - :py:obj:`calculate <pioran.acvf.Exponential.calculate>`\ (t)
        - Computes the exponential covariance function for an array of lags :math:`\tau`.


   .. rubric:: Members

   .. py:attribute:: parameters
      :type: pioran.parameters.ParametersModel

      
      Parameters of the covariance function.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: expression
      :value: 'exponential'

      
      Expression of the covariance function.
















      ..
          !! processed by numpydoc !!

   .. py:method:: calculate(t) -> jax.Array

      
      Computes the exponential covariance function for an array of lags :math:`\tau`.

      The expression is given by Equation :math:numref:`expocov`.
      with the variance :math:`A\ge 0` and length :math:`\gamma>0`.

      :Parameters:

          **t** : :obj:`jax.Array`
              Array of lags.

      :Returns:

          Covariance function evaluated on the array of lags.
              ..













      ..
          !! processed by numpydoc !!



.. py:class:: SquaredExponential(param_values: list[float], free_parameters: list[bool] = [True, True])

   Bases: :py:obj:`pioran.acvf_base.CovarianceFunction`

   
   Squared exponential covariance function.

   .. math:: :label: exposquare

       K(\tau) = A \times \exp{\left( -2 \pi^2 \tau^2 \sigma^2 \right)}.

   with the variance :math:`A\ge 0` and length :math:`\sigma>0`.

   The parameters are stored in the `parameters` attribute which is a :class:`~pioran.parameters.ParametersModel` object.
   The values of the parameters can be accessed using the `parameters` attribute via two keys: '`variance`' and '`length`'.

   The covariance function is evaluated on an array of lags :math:`\tau` using the `calculate` method.

   :Parameters:

       **param_values** : :obj:`list` of :obj:`float`
           Values of the parameters of the covariance function. [`variance`, `length`]

       **free_parameters** : :obj:`list` of :obj:`bool`
           List of bool to indicate if the parameters are free or not.














   ..
       !! processed by numpydoc !!

   .. rubric:: Overview

   .. list-table:: Attributes
      :header-rows: 0
      :widths: auto
      :class: summarytable

      * - :py:obj:`parameters <pioran.acvf.SquaredExponential.parameters>`
        - Parameters of the covariance function.
      * - :py:obj:`expression <pioran.acvf.SquaredExponential.expression>`
        - Expression of the covariance function.


   .. list-table:: Methods
      :header-rows: 0
      :widths: auto
      :class: summarytable

      * - :py:obj:`calculate <pioran.acvf.SquaredExponential.calculate>`\ (t)
        - Compute the squared exponential covariance function for an array of lags :math:`\tau`.


   .. rubric:: Members

   .. py:attribute:: parameters
      :type: pioran.parameters.ParametersModel

      
      Parameters of the covariance function.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: expression
      :value: 'squared_exponential'

      
      Expression of the covariance function.
















      ..
          !! processed by numpydoc !!

   .. py:method:: calculate(t) -> jax.Array

      
      Compute the squared exponential covariance function for an array of lags :math:`\tau`.

      The expression is given by Equation :math:numref:`exposquare`.
      with the variance :math:`A\ge 0` and length :math:`\sigma>0`.

      :Parameters:

          **t** : :obj:`jax.Array`
              Array of lags.

      :Returns:

          Covariance function evaluated on the array of lags.
              ..













      ..
          !! processed by numpydoc !!



.. py:class:: Matern32(param_values, free_parameters=[True, True])

   Bases: :py:obj:`pioran.acvf_base.CovarianceFunction`

   
   Matern 3/2 covariance function.

   .. math:: :label: matern32

      K(\tau) = A \times \left(1+\dfrac{ \sqrt{3} \tau}{\gamma} \right)  \exp{\left(-  \sqrt{3} |\tau| / \gamma \right)}.

   with the variance :math:`A\ge 0` and length :math:`\gamma>0`

   The parameters are stored in the `parameters` attribute which is a :class:`~pioran.parameters.ParametersModel` object.
   The values of the parameters can be accessed using the `parameters` attribute via two keys: '`variance`' and '`length`'.

   The covariance function is evaluated on an array of lags :math:`\tau` using the `calculate` method.

   :Parameters:

       **param_values** : :obj:`list` of :obj:`float`
           Values of the parameters of the covariance function. [`variance`, `length`]

       **free_parameters** : :obj:`list` of :obj:`bool`
           List of bool to indicate if the parameters are free or not.














   ..
       !! processed by numpydoc !!

   .. rubric:: Overview

   .. list-table:: Attributes
      :header-rows: 0
      :widths: auto
      :class: summarytable

      * - :py:obj:`parameters <pioran.acvf.Matern32.parameters>`
        - Parameters of the covariance function.
      * - :py:obj:`expression <pioran.acvf.Matern32.expression>`
        - Expression of the covariance function.


   .. list-table:: Methods
      :header-rows: 0
      :widths: auto
      :class: summarytable

      * - :py:obj:`calculate <pioran.acvf.Matern32.calculate>`\ (t)
        - Computes the Matérn 3/2 covariance function for an array of lags :math:`\tau`.


   .. rubric:: Members

   .. py:attribute:: parameters
      :type: pioran.parameters.ParametersModel

      
      Parameters of the covariance function.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: expression
      :value: 'matern32'

      
      Expression of the covariance function.
















      ..
          !! processed by numpydoc !!

   .. py:method:: calculate(t) -> jax.Array

      
      Computes the Matérn 3/2 covariance function for an array of lags :math:`\tau`.

      The expression is given by Equation :math:numref:`matern32`.
      with the variance :math:`A\ge 0` and scale :math:`\gamma>0`.

      :Parameters:

          **t** : :obj:`jax.Array`
              Array of lags.

      :Returns:

          Covariance function evaluated on the array of lags.
              ..













      ..
          !! processed by numpydoc !!



.. py:class:: Matern52(param_values, free_parameters=[True, True])

   Bases: :py:obj:`pioran.acvf_base.CovarianceFunction`

   
   Matern 5/2 covariance function.

   .. math:: :label: matern52

      K(\tau) = A \times \left(1+\dfrac{ \sqrt{5} \tau}{\gamma} + 5 \dfrac{\tau^2}{3\gamma^2} \right)  \exp{\left(-  \sqrt{5} |\tau| / \gamma \right) }.

   with the variance :math:`A\ge 0` and length :math:`\gamma>0`.

   The parameters are stored in the `parameters` attribute which is a :class:`~pioran.parameters.ParametersModel` object.
   The values of the parameters can be accessed using the `parameters` attribute via two keys: '`variance`' and '`length`'.

   The covariance function is evaluated on an array of lags :math:`\tau` using the `calculate` method.

   :Parameters:

       **param_values** : :obj:`list` of :obj:`float`
           Values of the parameters of the covariance function. [`variance`, `length`]

       **free_parameters** : :obj:`list` of :obj:`bool`
           List of bool to indicate if the parameters are free or not.














   ..
       !! processed by numpydoc !!

   .. rubric:: Overview

   .. list-table:: Attributes
      :header-rows: 0
      :widths: auto
      :class: summarytable

      * - :py:obj:`parameters <pioran.acvf.Matern52.parameters>`
        - Parameters of the covariance function.
      * - :py:obj:`expression <pioran.acvf.Matern52.expression>`
        - Expression of the covariance function.


   .. list-table:: Methods
      :header-rows: 0
      :widths: auto
      :class: summarytable

      * - :py:obj:`calculate <pioran.acvf.Matern52.calculate>`\ (t)
        - Computes the Matérn 5/2 covariance function for an array of lags :math:`\tau`.


   .. rubric:: Members

   .. py:attribute:: parameters
      :type: pioran.parameters.ParametersModel

      
      Parameters of the covariance function.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: expression
      :value: 'matern52'

      
      Expression of the covariance function.
















      ..
          !! processed by numpydoc !!

   .. py:method:: calculate(t) -> jax.Array

      
      Computes the Matérn 5/2 covariance function for an array of lags :math:`\tau`.

      The expression is given by Equation :math:numref:`matern52`.
      with the variance :math:`A\ge 0` and scale :math:`\gamma>0`.

      :Parameters:

          **t** : :obj:`jax.Array`
              Array of lags.

      :Returns:

          Covariance function evaluated on the array of lags.
              ..













      ..
          !! processed by numpydoc !!



.. py:class:: RationalQuadratic(param_values, free_parameters=[True, True, True])

   Bases: :py:obj:`pioran.acvf_base.CovarianceFunction`

   
   Rational quadratic covariance function.

   .. math:: :label: rationalquadratic

      K(\tau) = A \times {\left(1+ \dfrac{\tau^2}{2\alpha \gamma^2}  \right) }^{-\alpha}.

   with the variance :math:`A\ge 0`, length :math:`\gamma>0` and scale :math:`\alpha>0`

   The parameters are stored in the `parameters` attribute which is a :class:`~pioran.parameters.ParametersModel` object.
   The values of the parameters can be accessed using the `parameters` attribute via three keys: '`variance`', '`alpha`' and '`length`'.

   The covariance function is evaluated on an array of lags :math:`\tau` using the `calculate` method.

   :Parameters:

       **param_values** : :obj:`list` of :obj:`float`
           Values of the parameters of the covariance function. [`variance`, `alpha`, `length`]

       **free_parameters** : :obj:`list` of :obj:`bool`
           List of bool to indicate if the parameters are free or not.














   ..
       !! processed by numpydoc !!

   .. rubric:: Overview

   .. list-table:: Attributes
      :header-rows: 0
      :widths: auto
      :class: summarytable

      * - :py:obj:`parameters <pioran.acvf.RationalQuadratic.parameters>`
        - Parameters of the covariance function.
      * - :py:obj:`expression <pioran.acvf.RationalQuadratic.expression>`
        - Expression of the covariance function.


   .. list-table:: Methods
      :header-rows: 0
      :widths: auto
      :class: summarytable

      * - :py:obj:`calculate <pioran.acvf.RationalQuadratic.calculate>`\ (x)
        - Computes the rational quadratic covariance function for an array of lags :math:`\tau`.


   .. rubric:: Members

   .. py:attribute:: parameters
      :type: pioran.parameters.ParametersModel

      
      Parameters of the covariance function.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: expression
      :value: 'rationalquadratic'

      
      Expression of the covariance function.
















      ..
          !! processed by numpydoc !!

   .. py:method:: calculate(x) -> jax.Array

      
      Computes the rational quadratic covariance function for an array of lags :math:`\tau`.

      The expression is given by Equation :math:numref:`rationalquadratic`.
      with the variance :math:`A\ge 0`, length :math:`\gamma>0` and scale :math:`\alpha>0`.

      :Parameters:

          **t** : :obj:`jax.Array`
              Array of lags.

      :Returns:

          Covariance function evaluated on the array of lags.
              ..













      ..
          !! processed by numpydoc !!






