
parameters
==========

.. py:module:: pioran.parameters

.. autoapi-nested-parse::

   
   All the parameters of either :class:`~pioran.acvf_base.CovarianceFunction` or :class:`~pioran.psd_base.PowerSpectralDensity`
   are stored in a :class:`~pioran.parameters.ParametersModel` object which contains several instances of :class:`~pioran.parameters.Parameter`. 

   The base class for all parameters. By construction as a JAX Pytree_, some of the attributes are frozen and cannot be changed during runtime.
   The attributes that can be changed are the ``name`` and ``value`` of the parameter. The parameter can be considered as *free* or *fixed* depending on the value of the boolean attribute ``free``.
   A parameter can be referred to by its ``name`` or by index ``ID``. The ``ID`` is the index of the parameter in the list of parameters of the :class:`~pioran.parameters.ParametersModel` object.

   The attribute ``component`` is used to refer to parameters in individual components together when combining several model components.
   The attribute ``hyperparameter`` is used to refer to parameters that are not directly related to the model via covariance functions or power spectral densities.

   .. _Pytree: https://jax.readthedocs.io/en/latest/pytrees.html

   On top of the base class is built a :class:`~pioran.parameters.ParametersModel` object. This object inherits from :class:`equinox.Module`, which means
   the attributes of the :class:`~pioran.parameters.ParametersModel` object are immutable and cannot be changed during runtime.

   The values of the free parameters can be changed during runtime using the method :meth:`~pioran.parameters.ParametersModel.set_free_values`. 
   The names of the parameters can be changed using the method :meth:`~pioran.parameters.ParametersModel.set_names`.
   The values, names, IDs and free status of the parameters can be accessed using attributes. 

   The :class:`~pioran.parameters.Parameter` stored in :class:`~pioran.parameters.Parameter` can be accessed by the name of the parameter or by index with the ``[]`` operator. 
   If there are several parameters with the same name, the first one is returned.

   It is possible to add new parameters to the :class:`~pioran.parameters.ParametersModel` object using the method :meth:`~pioran.parameters.ParametersModel.append`.















   ..
       !! processed by numpydoc !!


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   parameter_base/index.rst
   parameters/index.rst


Overview
--------

.. list-table:: Classes
   :header-rows: 0
   :widths: auto
   :class: summarytable

   * - :py:obj:`Parameter <pioran.parameters.Parameter>`
     - Represents one parameter.
   * - :py:obj:`ParametersModel <pioran.parameters.ParametersModel>`
     - Stores the parameters of a model.




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

      * - :py:obj:`value <pioran.parameters.Parameter.value>`
        - Value of the parameter.
      * - :py:obj:`name <pioran.parameters.Parameter.name>`
        - Name of the parameter.
      * - :py:obj:`free <pioran.parameters.Parameter.free>`
        - If the parameter is free or fixed.
      * - :py:obj:`ID <pioran.parameters.Parameter.ID>`
        - ID of the parameter.
      * - :py:obj:`hyperparameter <pioran.parameters.Parameter.hyperparameter>`
        - If the parameter is an hyperparameter of the covariance function or not.
      * - :py:obj:`component <pioran.parameters.Parameter.component>`
        - Component containing the parameter.
      * - :py:obj:`relation <pioran.parameters.Parameter.relation>`
        - Relation between the parameter and the linked one.


   .. list-table:: Methods
      :header-rows: 0
      :widths: auto
      :class: summarytable

      * - :py:obj:`__str__ <pioran.parameters.Parameter.__str__>`\ ()
        - String representation of the parameter.
      * - :py:obj:`__repr__ <pioran.parameters.Parameter.__repr__>`\ ()
        - Return repr(self).
      * - :py:obj:`__repr_html__ <pioran.parameters.Parameter.__repr_html__>`\ ()
        - \-
      * - :py:obj:`tree_flatten <pioran.parameters.Parameter.tree_flatten>`\ ()
        - Flatten the object for the JAX tree.
      * - :py:obj:`tree_unflatten <pioran.parameters.Parameter.tree_unflatten>`\ (aux_data, children)
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



.. py:class:: ParametersModel(param_names: list[str], param_values: list[float], free_parameters: list[bool], IDs: list[int] = None, hyperparameters: list[bool] = None, components: list[int] = None, relations: list[str] = None, _pars: Union[None, list[pioran.parameters.parameter_base.Parameter]] = None)

   
   Stores the parameters of a model.

   Stores one or several :class:`~pioran.parameter_base.Parameter` objects for a model. The model can be
   of type :class:`~pioran.acvf_base.CovarianceFunction`  or :class:`~pioran.psd_base.PowerSpectralDensity`.
   Initialised with a list of values, names, free status for all parameters.

   :Parameters:

       **param_names** : :obj:`list` of :obj:`str`
           Names of the parameters.

       **param_values** : :obj:`list` of :obj:`float`
           Values of the parameters.

       **free_parameters** : :obj:`list` of :obj:`bool`, optional
           List of bool to indicate if the parameters are free or not.

       **IDs** : :obj:`list` of :obj:`int`, optional
           IDs of the parameters.

       **hyperparameters** : :obj:`list` of :obj:`bool`, optional
           List of bool to indicate if the parameters are hyperparameters or not.

       **components** : :obj:`list` of :obj:`int`, optional
           List of int to indicate the component number of the parameters.

       **relations** : :obj:`list` of :obj:`str`, optional
           List of str to indicate the relation between the parameters.

       **_pars** : :obj:`list` of :class:`~pioran.parameter_base.Parameter` or None, optional
           List of Parameter objects.














   ..
       !! processed by numpydoc !!

   .. rubric:: Overview

   .. list-table:: Attributes
      :header-rows: 0
      :widths: auto
      :class: summarytable

      * - :py:obj:`names <pioran.parameters.ParametersModel.names>`
        - Names of the parameters.
      * - :py:obj:`values <pioran.parameters.ParametersModel.values>`
        - Values of the parameters.
      * - :py:obj:`free_parameters <pioran.parameters.ParametersModel.free_parameters>`
        - True if the parameter is free, False otherwise.
      * - :py:obj:`components <pioran.parameters.ParametersModel.components>`
        - Component number of the parameters.
      * - :py:obj:`IDs <pioran.parameters.ParametersModel.IDs>`
        - IDs of the parameters.
      * - :py:obj:`hyperparameters <pioran.parameters.ParametersModel.hyperparameters>`
        - True if the parameter is a hyperparameter, False otherwise.
      * - :py:obj:`relations <pioran.parameters.ParametersModel.relations>`
        - Relation between the parameters.


   .. list-table:: Methods
      :header-rows: 0
      :widths: auto
      :class: summarytable

      * - :py:obj:`increment_component <pioran.parameters.ParametersModel.increment_component>`\ (increment)
        - Increment the component number of all the parameters by a given value.
      * - :py:obj:`increment_IDs <pioran.parameters.ParametersModel.increment_IDs>`\ (increment)
        - Increment the ID of all the parameters by a given value.
      * - :py:obj:`append <pioran.parameters.ParametersModel.append>`\ (name, value, free, ID, hyperparameter, component, relation)
        - Add a parameter to the list of objects.
      * - :py:obj:`set_names <pioran.parameters.ParametersModel.set_names>`\ (new_names)
        - Set the names of the parameters.
      * - :py:obj:`set_free_values <pioran.parameters.ParametersModel.set_free_values>`\ (new_free_values)
        - Set the values of the free parameters.
      * - :py:obj:`__getitem__ <pioran.parameters.ParametersModel.__getitem__>`\ (key)
        - Get a Parameter object using the name of the parameter in square brackets or the index of the parameter in brackets.
      * - :py:obj:`__len__ <pioran.parameters.ParametersModel.__len__>`\ ()
        - Get the number of parameters.
      * - :py:obj:`__setitem__ <pioran.parameters.ParametersModel.__setitem__>`\ (key, value)
        - Set a Parameter object using the name of the parameter in square brackets.
      * - :py:obj:`__str__ <pioran.parameters.ParametersModel.__str__>`\ ()
        - String representation of the Parameters object.
      * - :py:obj:`__repr__ <pioran.parameters.ParametersModel.__repr__>`\ ()
        - Return repr(self).
      * - :py:obj:`__repr_html__ <pioran.parameters.ParametersModel.__repr_html__>`\ ()
        - \-


   .. rubric:: Members

   .. py:attribute:: names
      :type: list[str]

      
      Names of the parameters.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: values
      :type: Union[list[float], jax.numpy.ndarray]

      
      Values of the parameters.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: free_parameters
      :type: list[bool]

      
      True if the parameter is free, False otherwise.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: components
      :type: list[int]

      
      Component number of the parameters.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: IDs
      :type: list[int]

      
      IDs of the parameters.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: hyperparameters
      :type: list[bool]

      
      True if the parameter is a hyperparameter, False otherwise.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: relations
      :type: list

      
      Relation between the parameters.
















      ..
          !! processed by numpydoc !!

   .. py:method:: increment_component(increment: int) -> None

      
      Increment the component number of all the parameters by a given value.


      :Parameters:

          **increment** : :obj:`int`
              Value used to increase the component number of the parameters.














      ..
          !! processed by numpydoc !!

   .. py:method:: increment_IDs(increment: int) -> None

      
      Increment the ID of all the parameters by a given value.


      :Parameters:

          **increment** : :obj:`int`
              Value used to increase the ID of the parameters.














      ..
          !! processed by numpydoc !!

   .. py:method:: append(name: str, value: float, free: bool, ID: int = None, hyperparameter: bool = True, component: int = None, relation=None) -> None

      
      Add a parameter to the list of objects.


      :Parameters:

          **name** : :obj:`str`
              Name of the parameter.

          **value** : `float`
              Value of the parameter.

          **free** : :obj:`bool`
              True if the parameter is free, False otherwise.

          **ID** : :obj:`int`, optional
              ID of the parameter.

          **hyperparameter** : :obj:`bool`, optional
              True if the parameter is a hyperparameter, False otherwise.
              The default is True.

          **component** : :obj:`int`, optional
              Component number of the parameter.

          **relation** : :obj:`str`, optional
              Relation between the parameters.














      ..
          !! processed by numpydoc !!

   .. py:method:: set_names(new_names: list[str]) -> None

      
      Set the names of the parameters.


      :Parameters:

          **new_names** : list of str
              New names of the parameters.





      :Raises:

          ValueError
              When the number of new names is not the same as the number of parameters.









      ..
          !! processed by numpydoc !!

   .. py:method:: set_free_values(new_free_values: list[float]) -> None

      
      Set the values of the free parameters.


      :Parameters:

          **new_free_values** : :obj:`list` of :obj:`float``
              Values of the free parameters.





      :Raises:

          ValueError
              When the number of new values is not the same as the number of free parameters.









      ..
          !! processed by numpydoc !!

   .. py:method:: __getitem__(key: Union[str, int]) -> pioran.parameters.parameter_base.Parameter

      
      Get a Parameter object using the name of the parameter in square brackets or the index of the parameter in brackets.

      Get the parameter object with the name in brackets : ['name'] or the parameter object with the index in brackets : [index].
      If several parameters have the same name, the only the first one is returned.

      :Parameters:

          **key** : :obj:`str` or :obj:`int`
              Name of the parameter or index of the parameter.

      :Returns:

          **parameter** : `Parameter` object
              Parameter with name "key".




      :Raises:

          KeyError
              When the parameter is not in the list of parameters.









      ..
          !! processed by numpydoc !!

   .. py:method:: __len__() -> int

      
      Get the number of parameters.



      :Returns:

          :obj:`int`
              Number of parameters.













      ..
          !! processed by numpydoc !!

   .. py:method:: __setitem__(key: str, value: pioran.parameters.parameter_base.Parameter) -> None

      
      Set a Parameter object using the name of the parameter in square brackets.


      :Parameters:

          **key** : :obj:`str`
              Name of the parameter.

          **value** : Parameter
              Value of the parameter with name "key".














      ..
          !! processed by numpydoc !!

   .. py:method:: __str__() -> str

      
      String representation of the Parameters object.



      :Returns:

          :obj:`str`
              Pretty table with the info on all parameters.













      ..
          !! processed by numpydoc !!

   .. py:method:: __repr__() -> str

      
      Return repr(self).
















      ..
          !! processed by numpydoc !!

   .. py:method:: __repr_html__() -> str







