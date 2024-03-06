
carma_acvf
==========

.. py:module:: pioran.carma.carma_acvf


Overview
--------

.. list-table:: Classes
   :header-rows: 0
   :widths: auto
   :class: summarytable

   * - :py:obj:`CARMA_covariance <pioran.carma.carma_acvf.CARMA_covariance>`
     - Covariance function of a Continuous AutoRegressive Moving Average (CARMA) process.




Classes
-------

.. py:class:: CARMA_covariance(p, q, AR_quad=None, MA_quad=None, beta=None, use_beta=True, lorentzian_centroids=None, lorentzian_widths=None, weights=None, **kwargs)

   Bases: :py:obj:`pioran.acvf.CovarianceFunction`

   
   Covariance function of a Continuous AutoRegressive Moving Average (CARMA) process.


   :Parameters:

       **p** : :obj:`int`
           Order of the AR part of the model.

       **q** : :obj:`int`
           Order of the MA part of the model. 0 <= q < p

       **AR_quad** : :obj:`list` of :obj:`float`
           Quadratic coefficients of the AR part of the model.

       **MA_quad** : :obj:`list` of :obj:`float`
           Quadratic coefficients of the MA part of the model.

       **beta** : :obj:`list` of :obj:`float`
           MA coefficients of the model.

       **use_beta** : :obj:`bool`
           If True, the MA coefficients are given by the beta parameters. If False, the MA coefficients are given by the quadratic coefficients.

       **lorentzian_centroids** : :obj:`list` of :obj:`float`
           Centroids of the Lorentzian functions.

       **lorentzian_widths** : :obj:`list` of :obj:`float`
           Widths of the Lorentzian functions.

       **weights** : :obj:`list` of :obj:`float`
           Weights of the Lorentzian functions.














   ..
       !! processed by numpydoc !!

   .. rubric:: Overview

   .. list-table:: Attributes
      :header-rows: 0
      :widths: auto
      :class: summarytable

      * - :py:obj:`parameters <pioran.carma.carma_acvf.CARMA_covariance.parameters>`
        - Parameters of the covariance function.
      * - :py:obj:`expression <pioran.carma.carma_acvf.CARMA_covariance.expression>`
        - Expression of the covariance function.
      * - :py:obj:`p <pioran.carma.carma_acvf.CARMA_covariance.p>`
        - Order of the AR part of the model.
      * - :py:obj:`q <pioran.carma.carma_acvf.CARMA_covariance.q>`
        - Order of the MA part of the model. 0 <= q < p
      * - :py:obj:`use_beta <pioran.carma.carma_acvf.CARMA_covariance.use_beta>`
        - If True, the MA coefficients are given by the beta parameters. If False, the MA coefficients are given by the quadratic coefficients.


   .. list-table:: Methods
      :header-rows: 0
      :widths: auto
      :class: summarytable

      * - :py:obj:`get_AR_quads <pioran.carma.carma_acvf.CARMA_covariance.get_AR_quads>`\ ()
        - Returns the quadratic coefficients of the AR part of the model.
      * - :py:obj:`get_MA_quads <pioran.carma.carma_acvf.CARMA_covariance.get_MA_quads>`\ ()
        - Returns the quadratic coefficients of the MA part of the model.
      * - :py:obj:`get_MA_coeffs <pioran.carma.carma_acvf.CARMA_covariance.get_MA_coeffs>`\ ()
        - Returns the quadratic coefficients of the AR part of the model.
      * - :py:obj:`get_AR_roots <pioran.carma.carma_acvf.CARMA_covariance.get_AR_roots>`\ ()
        - Returns the roots of the AR part of the model.
      * - :py:obj:`calculate <pioran.carma.carma_acvf.CARMA_covariance.calculate>`\ (tau)
        - Compute the autocovariance function of a CARMA(p,q) process.


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

   .. py:attribute:: p
      :type: int

      
      Order of the AR part of the model.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: q
      :type: int

      
      Order of the MA part of the model. 0 <= q < p
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: use_beta
      :type: bool

      
      If True, the MA coefficients are given by the beta parameters. If False, the MA coefficients are given by the quadratic coefficients.
















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

   .. py:method:: calculate(tau: jax.Array) -> jax.Array

      
      Compute the autocovariance function of a CARMA(p,q) process.
















      ..
          !! processed by numpydoc !!






