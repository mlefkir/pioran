
acvf_base
=========

.. py:module:: pioran.acvf_base

.. autoapi-nested-parse::

   Base representation of a covariance function. It is not meant to be used directly, but rather as a base class to build covariance functions. 
   The sum and product of covariance functions are implemented with the ``+`` and ``*`` operators, respectively.

   ..
       !! processed by numpydoc !!


Overview
--------

.. list-table:: Classes
   :header-rows: 0
   :widths: auto
   :class: summarytable

   * - :py:obj:`CovarianceFunction <pioran.acvf_base.CovarianceFunction>`
     - Represents a covariance function model.
   * - :py:obj:`ProductCovarianceFunction <pioran.acvf_base.ProductCovarianceFunction>`
     - Represents the product of two covariance functions.
   * - :py:obj:`SumCovarianceFunction <pioran.acvf_base.SumCovarianceFunction>`
     - Represents the sum of two covariance functions.




Classes
-------

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

      * - :py:obj:`parameters <pioran.acvf_base.CovarianceFunction.parameters>`
        - Parameters of the covariance function.
      * - :py:obj:`expression <pioran.acvf_base.CovarianceFunction.expression>`
        - Expression of the covariance function.


   .. list-table:: Methods
      :header-rows: 0
      :widths: auto
      :class: summarytable

      * - :py:obj:`__str__ <pioran.acvf_base.CovarianceFunction.__str__>`\ ()
        - String representation of the covariance function.
      * - :py:obj:`__repr__ <pioran.acvf_base.CovarianceFunction.__repr__>`\ ()
        - Representation of the covariance function.
      * - :py:obj:`get_cov_matrix <pioran.acvf_base.CovarianceFunction.get_cov_matrix>`\ (xq, xp)
        - Compute the covariance matrix between two arrays xq, xp.
      * - :py:obj:`__add__ <pioran.acvf_base.CovarianceFunction.__add__>`\ (other)
        - Overload of the + operator to add two covariance functions.
      * - :py:obj:`__mul__ <pioran.acvf_base.CovarianceFunction.__mul__>`\ (other)
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



.. py:class:: ProductCovarianceFunction(cov1: CovarianceFunction, cov2: CovarianceFunction)

   Bases: :py:obj:`CovarianceFunction`

   
   Represents the product of two covariance functions.


   :Parameters:

       **cov1** : :obj:`CovarianceFunction`
           First covariance function.

       **cov2** : :obj:`CovarianceFunction`
           Second covariance function.














   ..
       !! processed by numpydoc !!

   .. rubric:: Overview

   .. list-table:: Attributes
      :header-rows: 0
      :widths: auto
      :class: summarytable

      * - :py:obj:`cov1 <pioran.acvf_base.ProductCovarianceFunction.cov1>`
        - First covariance function.
      * - :py:obj:`cov2 <pioran.acvf_base.ProductCovarianceFunction.cov2>`
        - Second covariance function.
      * - :py:obj:`parameters <pioran.acvf_base.ProductCovarianceFunction.parameters>`
        - Parameters of the covariance function.
      * - :py:obj:`expression <pioran.acvf_base.ProductCovarianceFunction.expression>`
        - Expression of the total covariance function.


   .. list-table:: Methods
      :header-rows: 0
      :widths: auto
      :class: summarytable

      * - :py:obj:`calculate <pioran.acvf_base.ProductCovarianceFunction.calculate>`\ (x)
        - Compute the covariance function at the points x.


   .. rubric:: Members

   .. py:attribute:: cov1
      :type: CovarianceFunction

      
      First covariance function.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: cov2
      :type: CovarianceFunction

      
      Second covariance function.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: parameters
      :type: pioran.parameters.ParametersModel

      
      Parameters of the covariance function.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: expression
      :type: str

      
      Expression of the total covariance function.
















      ..
          !! processed by numpydoc !!

   .. py:method:: calculate(x: jax.Array) -> jax.Array

      
      Compute the covariance function at the points x.

      It is the product of the two covariance functions.

      :Parameters:

          **x** : :obj:`jax.Array`
              Points where the covariance function is computed.

      :Returns:

          Product of the two covariance functions at the points x.
              ..













      ..
          !! processed by numpydoc !!



.. py:class:: SumCovarianceFunction(cov1: CovarianceFunction, cov2: CovarianceFunction)

   Bases: :py:obj:`CovarianceFunction`

   
   Represents the sum of two covariance functions.


   :Parameters:

       **cov1** : :obj:`CovarianceFunction`
           First covariance function.

       **cov2** : :obj:`CovarianceFunction`
           Second covariance function.














   ..
       !! processed by numpydoc !!

   .. rubric:: Overview

   .. list-table:: Attributes
      :header-rows: 0
      :widths: auto
      :class: summarytable

      * - :py:obj:`cov1 <pioran.acvf_base.SumCovarianceFunction.cov1>`
        - First covariance function.
      * - :py:obj:`cov2 <pioran.acvf_base.SumCovarianceFunction.cov2>`
        - Second covariance function.
      * - :py:obj:`parameters <pioran.acvf_base.SumCovarianceFunction.parameters>`
        - Parameters of the covariance function.
      * - :py:obj:`expression <pioran.acvf_base.SumCovarianceFunction.expression>`
        - Expression of the total covariance function.


   .. list-table:: Methods
      :header-rows: 0
      :widths: auto
      :class: summarytable

      * - :py:obj:`calculate <pioran.acvf_base.SumCovarianceFunction.calculate>`\ (x)
        - Compute the covariance function at the points x.


   .. rubric:: Members

   .. py:attribute:: cov1
      :type: CovarianceFunction

      
      First covariance function.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: cov2
      :type: CovarianceFunction

      
      Second covariance function.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: parameters
      :type: pioran.parameters.ParametersModel

      
      Parameters of the covariance function.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: expression
      :type: str

      
      Expression of the total covariance function.
















      ..
          !! processed by numpydoc !!

   .. py:method:: calculate(x: jax.Array) -> jax.Array

      
      Compute the covariance function at the points x.

      It is the sum of the two covariance functions.

      :Parameters:

          **x** : :obj:`jax.Array`
              Points where the covariance function is computed.

      :Returns:

          :obj:`SumCovarianceFunction`
              Sum of the two covariance functions at the points x.













      ..
          !! processed by numpydoc !!






