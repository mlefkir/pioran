
carma_model
===========

.. py:module:: pioran.carma.carma_model


Overview
--------

.. list-table:: Classes
   :header-rows: 0
   :widths: auto
   :class: summarytable

   * - :py:obj:`CARMA_model <pioran.carma.carma_model.CARMA_model>`
     - Base class for Continuous-time AutoRegressive Moving Average (CARMA) models. Inherits from eqxinox.Module.




Classes
-------

.. py:class:: CARMA_model(p, q, AR_quad=None, MA_quad=None, beta=None, use_beta=True, lorentzian_centroids=None, lorentzian_widths=None, weights=None, **kwargs)

   Bases: :py:obj:`equinox.Module`

   
   Base class for Continuous-time AutoRegressive Moving Average (CARMA) models. Inherits from eqxinox.Module.

   This class implements the basic functionality for CARMA models

   :Parameters:

       **parameters** : :obj:`ParametersModel`
           Parameters of the model.

       **p** : :obj:`int`
           Order of the AR part of the model.

       **q** : :obj:`int`
           Order of the MA part of the model. 0 <= q < p












   :Attributes:

       **parameters** : :obj:`ParametersModel`
           Parameters of the model.

       **p** : :obj:`int`
           Order of the AR part of the model.

       **q** : :obj:`int`
           Order of the MA part of the model. 0 <= q < p

       **_p** : :obj:`int`
           Order of the AR part of the model. p+1

       **_q** : :obj:`int`
           Order of the MA part of the model. q+1   


   ..
       !! processed by numpydoc !!

   .. rubric:: Overview

   .. list-table:: Attributes
      :header-rows: 0
      :widths: auto
      :class: summarytable

      * - :py:obj:`parameters <pioran.carma.carma_model.CARMA_model.parameters>`
        - \-
      * - :py:obj:`ndims <pioran.carma.carma_model.CARMA_model.ndims>`
        - \-
      * - :py:obj:`p <pioran.carma.carma_model.CARMA_model.p>`
        - \-
      * - :py:obj:`q <pioran.carma.carma_model.CARMA_model.q>`
        - \-
      * - :py:obj:`use_beta <pioran.carma.carma_model.CARMA_model.use_beta>`
        - \-


   .. list-table:: Methods
      :header-rows: 0
      :widths: auto
      :class: summarytable

      * - :py:obj:`__str__ <pioran.carma.carma_model.CARMA_model.__str__>`\ ()
        - String representation of the model.
      * - :py:obj:`__repr__ <pioran.carma.carma_model.CARMA_model.__repr__>`\ ()
        - Return repr(self).
      * - :py:obj:`PowerSpectrum <pioran.carma.carma_model.CARMA_model.PowerSpectrum>`\ (f)
        - Computes the power spectrum of the CARMA process.
      * - :py:obj:`get_AR_quads <pioran.carma.carma_model.CARMA_model.get_AR_quads>`\ ()
        - Returns the quadratic coefficients of the AR part of the model.
      * - :py:obj:`get_MA_quads <pioran.carma.carma_model.CARMA_model.get_MA_quads>`\ ()
        - Returns the quadratic coefficients of the MA part of the model.
      * - :py:obj:`get_AR_coeffs <pioran.carma.carma_model.CARMA_model.get_AR_coeffs>`\ ()
        - Returns the coefficients of the AR part of the model.
      * - :py:obj:`get_MA_coeffs <pioran.carma.carma_model.CARMA_model.get_MA_coeffs>`\ ()
        - Returns the quadratic coefficients of the AR part of the model.
      * - :py:obj:`get_AR_roots <pioran.carma.carma_model.CARMA_model.get_AR_roots>`\ ()
        - Returns the roots of the AR part of the model.
      * - :py:obj:`Autocovariance <pioran.carma.carma_model.CARMA_model.Autocovariance>`\ (tau)
        - Compute the autocovariance function of a CARMA(p,q) process.
      * - :py:obj:`init_statespace <pioran.carma.carma_model.CARMA_model.init_statespace>`\ (y_0, errsize)
        - Initialises the state space representation of the model
      * - :py:obj:`statespace_representation <pioran.carma.carma_model.CARMA_model.statespace_representation>`\ (dt)
        - \-


   .. rubric:: Members

   .. py:attribute:: parameters
      :type: pioran.parameters.ParametersModel

      

   .. py:attribute:: ndims
      :type: int

      

   .. py:attribute:: p
      :type: int

      

   .. py:attribute:: q
      :type: int

      

   .. py:attribute:: use_beta
      :type: bool

      

   .. py:method:: __str__() -> str

      
      String representation of the model.

      Also prints the roots and coefficients of the AR and MA parts of the model.















      ..
          !! processed by numpydoc !!

   .. py:method:: __repr__() -> str

      
      Return repr(self).
















      ..
          !! processed by numpydoc !!

   .. py:method:: PowerSpectrum(f: jax.Array) -> jax.Array

      
      Computes the power spectrum of the CARMA process.


      :Parameters:

          **f** : :obj:`jax.Array`
              Frequencies at which the power spectrum is evaluated.

      :Returns:

          **P** : :obj:`jax.Array`
              Power spectrum of the CARMA process.













      ..
          !! processed by numpydoc !!

   .. py:method:: get_AR_quads() -> jax.Array

      
      Returns the quadratic coefficients of the AR part of the model.

      Iterates over the parameters of the model and returns the quadratic
      coefficients of the AR part of the model.


      :Returns:

          :obj:`jax.Array`
              Quadratic coefficients of the AR part of the model.













      ..
          !! processed by numpydoc !!

   .. py:method:: get_MA_quads() -> jax.Array

      
      Returns the quadratic coefficients of the MA part of the model.

      Iterates over the parameters of the model and returns the quadratic
      coefficients of the MA part of the model.


      :Returns:

          :obj:`jax.Array`
              Quadratic coefficients of the MA part of the model.













      ..
          !! processed by numpydoc !!

   .. py:method:: get_AR_coeffs() -> jax.Array

      
      Returns the coefficients of the AR part of the model.



      :Returns:

          :obj:`jax.Array`
              Coefficients of the AR part of the model.













      ..
          !! processed by numpydoc !!

   .. py:method:: get_MA_coeffs() -> jax.Array

      
      Returns the quadratic coefficients of the AR part of the model.

      Iterates over the parameters of the model and returns the quadratic
      coefficients of the AR part of the model.


      :Returns:

          :obj:`jax.Array`
              Quadratic coefficients of the AR part of the model.













      ..
          !! processed by numpydoc !!

   .. py:method:: get_AR_roots() -> jax.Array

      
      Returns the roots of the AR part of the model.



      :Returns:

          :obj:`jax.Array`
              Roots of the AR part of the model.













      ..
          !! processed by numpydoc !!

   .. py:method:: Autocovariance(tau: jax.Array) -> jax.Array

      
      Compute the autocovariance function of a CARMA(p,q) process.
















      ..
          !! processed by numpydoc !!

   .. py:method:: init_statespace(y_0=None, errsize=None) -> Tuple[jax.Array, jax.Array] | Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]

      
      Initialises the state space representation of the model

      Parameters















      ..
          !! processed by numpydoc !!

   .. py:method:: statespace_representation(dt: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array] | jax.Array







