
parameters
==========

.. py:module:: pioran.parameters.parameters

.. autoapi-nested-parse::

   On top of the base class is built a :class:`~pioran.parameters.ParametersModel` object. This object inherits from :class:`equinox.Module`, which means
   the attributes of the :class:`~pioran.parameters.ParametersModel` object at immutable and cannot be changed during runtime.

   The values of the free parameters can be changed during runtime using the method :meth:`~pioran.parameters.ParametersModel.set_free_values`. 
   The names of the parameters can be changed using the method :meth:`~pioran.parameters.ParametersModel.set_names`.
   The values, names, IDs and free status of the parameters can be accessed using attributes. 

   The :class:`~pioran.parameter_base.Parameter` stored in :class:`~pioran.parameter_base.Parameter` can be accessed by the name of the parameter or by index with the ``[]`` operator. 
   If there are several parameters with the same name, the first one is returned.

   It is possible to add new parameters to the :class:`~pioran.parameters.ParametersModel` object using the method :meth:`~pioran.parameters.ParametersModel.append`.

   ..
       !! processed by numpydoc !!


Overview
--------

.. list-table:: Classes
   :header-rows: 0
   :widths: auto
   :class: summarytable

   * - :py:obj:`ParametersModel <pioran.parameters.parameters.ParametersModel>`
     - Stores the parameters of a model.




Classes
-------

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

      * - :py:obj:`names <pioran.parameters.parameters.ParametersModel.names>`
        - Names of the parameters.
      * - :py:obj:`values <pioran.parameters.parameters.ParametersModel.values>`
        - Values of the parameters.
      * - :py:obj:`free_parameters <pioran.parameters.parameters.ParametersModel.free_parameters>`
        - True if the parameter is free, False otherwise.
      * - :py:obj:`components <pioran.parameters.parameters.ParametersModel.components>`
        - Component number of the parameters.
      * - :py:obj:`IDs <pioran.parameters.parameters.ParametersModel.IDs>`
        - IDs of the parameters.
      * - :py:obj:`hyperparameters <pioran.parameters.parameters.ParametersModel.hyperparameters>`
        - True if the parameter is a hyperparameter, False otherwise.
      * - :py:obj:`relations <pioran.parameters.parameters.ParametersModel.relations>`
        - Relation between the parameters.


   .. list-table:: Methods
      :header-rows: 0
      :widths: auto
      :class: summarytable

      * - :py:obj:`increment_component <pioran.parameters.parameters.ParametersModel.increment_component>`\ (increment)
        - Increment the component number of all the parameters by a given value.
      * - :py:obj:`increment_IDs <pioran.parameters.parameters.ParametersModel.increment_IDs>`\ (increment)
        - Increment the ID of all the parameters by a given value.
      * - :py:obj:`append <pioran.parameters.parameters.ParametersModel.append>`\ (name, value, free, ID, hyperparameter, component, relation)
        - Add a parameter to the list of objects.
      * - :py:obj:`set_names <pioran.parameters.parameters.ParametersModel.set_names>`\ (new_names)
        - Set the names of the parameters.
      * - :py:obj:`set_free_values <pioran.parameters.parameters.ParametersModel.set_free_values>`\ (new_free_values)
        - Set the values of the free parameters.
      * - :py:obj:`__getitem__ <pioran.parameters.parameters.ParametersModel.__getitem__>`\ (key)
        - Get a Parameter object using the name of the parameter in square brackets or the index of the parameter in brackets.
      * - :py:obj:`__len__ <pioran.parameters.parameters.ParametersModel.__len__>`\ ()
        - Get the number of parameters.
      * - :py:obj:`__setitem__ <pioran.parameters.parameters.ParametersModel.__setitem__>`\ (key, value)
        - Set a Parameter object using the name of the parameter in square brackets.
      * - :py:obj:`__str__ <pioran.parameters.parameters.ParametersModel.__str__>`\ ()
        - String representation of the Parameters object.
      * - :py:obj:`__repr__ <pioran.parameters.parameters.ParametersModel.__repr__>`\ ()
        - Return repr(self).
      * - :py:obj:`__repr_html__ <pioran.parameters.parameters.ParametersModel.__repr_html__>`\ ()
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







