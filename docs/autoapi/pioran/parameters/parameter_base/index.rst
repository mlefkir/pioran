
parameter_base
==============

.. py:module:: pioran.parameters.parameter_base

.. autoapi-nested-parse::

   All the parameters of either :class:`~pioran.acvf_base.CovarianceFunction` or :class:`~pioran.psd_base.PowerSpectralDensity`
   are stored in a :class:`~pioran.parameters.ParametersModel` object which contains several instances of :class:`~pioran.parameter_base.Parameter`. 

   The base class for all parameters. By construction as a JAX Pytree_, some of the attributes are frozen and cannot be changed during runtime.
   The attributes that can be changed are the ``name`` and ``value`` of the parameter. The parameter can be considered as *free* or *fixed* depending on the value of the boolean attribute ``free``.
   A parameter can be referred to by its ``name`` or by index ``ID``. The ``ID`` is the index of the parameter in the list of parameters of the :class:`~pioran.parameters.ParametersModel` object.

   The attribute ``component`` is used to refer to parameters in individual components together when combining several model components.
   The attribute ``hyperparameter`` is used to refer to parameters that are not directly related to the model via covariance functions or power spectral densities.

   .. _Pytree: https://jax.readthedocs.io/en/latest/pytrees.html

   ..
       !! processed by numpydoc !!


Overview
--------

.. list-table:: Classes
   :header-rows: 0
   :widths: auto
   :class: summarytable

   * - :py:obj:`Parameter <pioran.parameters.parameter_base.Parameter>`
     - Represents one parameter.




Classes
-------

.. py:class:: Parameter(name: str, value: float, free: bool = True, ID: int = 1, hyperparameter: bool = True, component=1, relation=None)

   
   Represents one parameter.

   The object used to create a list of
   parameters with the :class:`~pioran.parameters.ParametersModel` object.

   :Parameters:

       **name** : :obj:`str`
           Name of the parameter.

       **value** : :obj:`float`
           Value of the parameter.

       **ID** : :obj:`int`, optional
           ID of the parameter, default is 1.

       **free** : :obj:`bool`
           If the parameter is free or fixed.

       **hyperparameter** : :obj:`bool`, optional
           If the parameter is an hyperparameter of the covariance function or not. The default is True.

       **component** : :obj:`int`, optional
           Component containing the parameter, default is 1.

       **relation** : :obj:`Parameter`, optional
           Relation between the parameter and the linked one. The default is None.














   ..
       !! processed by numpydoc !!

   .. rubric:: Overview

   .. list-table:: Attributes
      :header-rows: 0
      :widths: auto
      :class: summarytable

      * - :py:obj:`value <pioran.parameters.parameter_base.Parameter.value>`
        - Value of the parameter.
      * - :py:obj:`name <pioran.parameters.parameter_base.Parameter.name>`
        - Name of the parameter.
      * - :py:obj:`free <pioran.parameters.parameter_base.Parameter.free>`
        - If the parameter is free or fixed.
      * - :py:obj:`ID <pioran.parameters.parameter_base.Parameter.ID>`
        - ID of the parameter.
      * - :py:obj:`hyperparameter <pioran.parameters.parameter_base.Parameter.hyperparameter>`
        - If the parameter is an hyperparameter of the covariance function or not.
      * - :py:obj:`component <pioran.parameters.parameter_base.Parameter.component>`
        - Component containing the parameter.
      * - :py:obj:`relation <pioran.parameters.parameter_base.Parameter.relation>`
        - Relation between the parameter and the linked one.


   .. list-table:: Methods
      :header-rows: 0
      :widths: auto
      :class: summarytable

      * - :py:obj:`__str__ <pioran.parameters.parameter_base.Parameter.__str__>`\ ()
        - String representation of the parameter.
      * - :py:obj:`__repr__ <pioran.parameters.parameter_base.Parameter.__repr__>`\ ()
        - Return repr(self).
      * - :py:obj:`__repr_html__ <pioran.parameters.parameter_base.Parameter.__repr_html__>`\ ()
        - \-
      * - :py:obj:`tree_flatten <pioran.parameters.parameter_base.Parameter.tree_flatten>`\ ()
        - Flatten the object for the JAX tree.
      * - :py:obj:`tree_unflatten <pioran.parameters.parameter_base.Parameter.tree_unflatten>`\ (aux_data, children)
        - :summarylabel:`class` Unflatten the object for the JAX tree.


   .. rubric:: Members

   .. py:attribute:: value
      :type: float

      
      Value of the parameter.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: name
      :type: str

      
      Name of the parameter.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: free
      :type: bool

      
      If the parameter is free or fixed.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: ID
      :type: int

      
      ID of the parameter.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: hyperparameter
      :type: bool

      
      If the parameter is an hyperparameter of the covariance function or not.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: component
      :type: int

      
      Component containing the parameter.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: relation
      :type: None

      
      Relation between the parameter and the linked one.
















      ..
          !! processed by numpydoc !!

   .. py:method:: __str__() -> str

      
      String representation of the parameter.

      In the following format:

      component  ID  name  value  free  linked  type


      :Returns:

          :obj:`str`
              String representation of the parameter.













      ..
          !! processed by numpydoc !!

   .. py:method:: __repr__() -> str

      
      Return repr(self).
















      ..
          !! processed by numpydoc !!

   .. py:method:: __repr_html__() -> str


   .. py:method:: tree_flatten()

      
      Flatten the object for the JAX tree.

      The object is flatten in a tuple containing the dynamic children and the static auxiliary data.
      The dynamic children are the :py:attr:`name` and :py:attr:`value` of the parameter while the static auxiliary data are the attributes
      :py:attr:`free`, :py:attr:`ID`, :py:attr:`hyperparameter`, :py:attr:`component` and :py:attr:`relation`.


      :Returns:

          :obj:`tuple`
              Tuple containing the children and the auxiliary data.













      ..
          !! processed by numpydoc !!

   .. py:method:: tree_unflatten(aux_data: dict, children: tuple)
      :classmethod:

      
      Unflatten the object for the JAX tree.


      :Parameters:

          **aux_data** : :obj:`dict`
              Dictionary containing the static auxiliary data.

          **children** : :obj:`tuple`
              Tuple containing the dynamic children.

      :Returns:

          :obj:`Parameter`
              Parameter object.













      ..
          !! processed by numpydoc !!






