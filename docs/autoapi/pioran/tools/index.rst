
tools
=====

.. py:module:: pioran.tools

.. autoapi-nested-parse::

   Various tools for the Gaussian Process module.

   ..
       !! processed by numpydoc !!


Overview
--------


.. list-table:: Function
   :header-rows: 0
   :widths: auto
   :class: summarytable

   * - :py:obj:`sanity_checks <pioran.tools.sanity_checks>`\ (array_A, array_B)
     - Check if the arrays are of the same shape
   * - :py:obj:`reshape_array <pioran.tools.reshape_array>`\ (array)
     - Reshape the array to a 2D array with jnp.shape(array,(len(array),1).


.. list-table:: Attributes
   :header-rows: 0
   :widths: auto
   :class: summarytable

   * - :py:obj:`TYPE_NUMBER <pioran.tools.TYPE_NUMBER>`
     - \-
   * - :py:obj:`Array_type <pioran.tools.Array_type>`
     - \-
   * - :py:obj:`TABLE_LENGTH <pioran.tools.TABLE_LENGTH>`
     - \-
   * - :py:obj:`HEADER_PARAMETERS <pioran.tools.HEADER_PARAMETERS>`
     - \-



Functions
---------
.. py:function:: sanity_checks(array_A, array_B)

   
   Check if the arrays are of the same shape 


   :Parameters:

       **array_A: (n,1) :obj:`jax.Array`**
           First array.

       **array_B: (n,1) :obj:`jax.Array`**
           Second array.














   ..
       !! processed by numpydoc !!

.. py:function:: reshape_array(array)

   
   Reshape the array to a 2D array with jnp.shape(array,(len(array),1).


   :Parameters:

       **array: (n,) :obj:`jax.Array`**
           ..

   :Returns:

       array: (n,1) :obj:`jax.Array`
           Reshaped array.













   ..
       !! processed by numpydoc !!


Attributes
----------
.. py:data:: TYPE_NUMBER

   

.. py:data:: Array_type

   

.. py:data:: TABLE_LENGTH
   :value: 76

   

.. py:data:: HEADER_PARAMETERS
   :value: '{Component:<4} {ID:<4} {Name:<15} {Value:<14} {Status:<9} {Linked:<9} {Type:<15} '

   



