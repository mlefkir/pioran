
carma_utils
===========

.. py:module:: pioran.carma.carma_utils


Overview
--------


.. list-table:: Function
   :header-rows: 0
   :widths: auto
   :class: summarytable

   * - :py:obj:`initialise_CARMA_object <pioran.carma.carma_utils.initialise_CARMA_object>`\ (self, p, q, AR_quad, MA_quad, beta, use_beta, lorentzian_centroids, lorentzian_widths, weights, \*\*kwargs)
     - Function to initialise the CARMA object for either the CARMA model object or the CARMA covariance function object.
   * - :py:obj:`quad_to_roots <pioran.carma.carma_utils.quad_to_roots>`\ (quad)
     - Convert the coefficients of the quadratic form to coefficients of the AR polynomial.
   * - :py:obj:`roots_to_quad <pioran.carma.carma_utils.roots_to_quad>`\ (roots)
     - Convert the roots of the AR polynomial to the coefficients quad of the quadratic polynomial.
   * - :py:obj:`MA_quad_to_coeff <pioran.carma.carma_utils.MA_quad_to_coeff>`\ (q, quad)
     - Convert the coefficients quad of the quadratic polynomial to the coefficients alpha of the AR polynomial
   * - :py:obj:`quad_to_coeff <pioran.carma.carma_utils.quad_to_coeff>`\ (quad)
     - Convert the coefficients quad of the quadratic polynomial to the coefficients alpha of the AR polynomial
   * - :py:obj:`lorentzians_to_roots <pioran.carma.carma_utils.lorentzians_to_roots>`\ (Widths, Centroids)
     - Convert the widths and centroids of the Lorentzian functions to the roots of the AR polynomial.
   * - :py:obj:`get_U <pioran.carma.carma_utils.get_U>`\ (roots_AR)
     - \-
   * - :py:obj:`get_V <pioran.carma.carma_utils.get_V>`\ (J, roots_AR)
     - \-
   * - :py:obj:`CARMA_powerspectrum <pioran.carma.carma_utils.CARMA_powerspectrum>`\ (f, alpha, beta, sigma)
     - Computes the power spectrum of the CARMA process.
   * - :py:obj:`CARMA_autocovariance <pioran.carma.carma_utils.CARMA_autocovariance>`\ (tau, roots_AR, beta, sigma)
     - Computes the autocovariance of the CARMA process.




Functions
---------
.. py:function:: initialise_CARMA_object(self, p, q, AR_quad=None, MA_quad=None, beta=None, use_beta=True, lorentzian_centroids=None, lorentzian_widths=None, weights=None, **kwargs) -> None

   
   Function to initialise the CARMA object for either the CARMA model object or the CARMA covariance function object.

   This function is used in the constructor of the CARMA model object and the CARMA covariance function object.

   :Parameters:

       **self** : :obj:`CARMA_model` or :obj:`CARMA_covariance_function`
           The CARMA model or covariance function object.

       **p** : :obj:`int`
           Order of the AR polynomial.

       **q** : :obj:`int`
           Order of the MA polynomial.

       **AR_quad** : :obj:`Array_type`
           Quadratic coefficients of the AR polynomial, None if not provided.

       **MA_quad** : :obj:`Array_type`
           Quadratic coefficients of the MA polynomial, None if not provided.

       **beta** : :obj:`Array_type`
           Weights of the MA polynomial, None if not provided.

       **use_beta** : :obj:`bool`
           If True, the MA polynomial is parametrised by the weights beta, otherwise it is parametrised by the quadratic coefficients MA_quad.

       **lorentzian_centroids** : :obj:`Array_type`
           Centroids of the Lorentzian components of the covariance function, None if not provided.

       **lorentzian_widths** : :obj:`Array_type`
           Widths of the Lorentzian components of the covariance function, None if not provided.

       **weights** : :obj:`Array_type`
           Weights of the Lorentzian components of the covariance function, None if not provided, equivalent to beta.

       **\*\*kwargs** : :obj:`dict`
           Keyword arguments for the CARMA model.














   ..
       !! processed by numpydoc !!

.. py:function:: quad_to_roots(quad: jax.Array) -> jax.Array

   
   Convert the coefficients of the quadratic form to coefficients of the AR polynomial.
















   ..
       !! processed by numpydoc !!

.. py:function:: roots_to_quad(roots: jax.Array) -> jax.Array

   
   Convert the roots of the AR polynomial to the coefficients quad of the quadratic polynomial.
















   ..
       !! processed by numpydoc !!

.. py:function:: MA_quad_to_coeff(q, quad: jax.Array) -> jax.Array

   
   Convert the coefficients quad of the quadratic polynomial to the coefficients alpha of the AR polynomial


   :Parameters:

       **quad** : :obj:`jax.Array`
           Coefficients quad of the quadratic polynomial

   :Returns:

       :obj:`jax.Array`
           Coefficients alpha of the AR polynomial













   ..
       !! processed by numpydoc !!

.. py:function:: quad_to_coeff(quad: jax.Array) -> jax.Array

   
   Convert the coefficients quad of the quadratic polynomial to the coefficients alpha of the AR polynomial


   :Parameters:

       **quad** : :obj:`jax.Array`
           Coefficients quad of the quadratic polynomial

   :Returns:

       :obj:`jax.Array`
           Coefficients alpha of the AR polynomial













   ..
       !! processed by numpydoc !!

.. py:function:: lorentzians_to_roots(Widths: jax.Array, Centroids: jax.Array) -> jax.Array

   
   Convert the widths and centroids of the Lorentzian functions to the roots of the AR polynomial.


   :Parameters:

       **Widths** : :obj:`jax.Array`
           Widths of the Lorentzian functions.

       **Centroids** : :obj:`jax.Array`
           Centroids of the Lorentzian functions.

   :Returns:

       :obj:`jax.Array`
           Roots of the AR polynomial.













   ..
       !! processed by numpydoc !!

.. py:function:: get_U(roots_AR: jax.Array) -> jax.Array


.. py:function:: get_V(J: jax.Array, roots_AR: jax.Array) -> jax.Array


.. py:function:: CARMA_powerspectrum(f: jax.Array, alpha, beta, sigma) -> jax.Array

   
   Computes the power spectrum of the CARMA process.


   :Parameters:

       **f** : :obj:`jax.Array`
           Frequencies at which the power spectrum is evaluated.

       **alpha** : :obj:`jax.Array`
           Coefficients of the AR polynomial.

       **beta** : :obj:`jax.Array`
           Coefficients of the MA polynomial.

       **sigma** : :obj:`float`
           Standard deviation of the white noise process.

   :Returns:

       **P** : :obj:`jax.Array`
           Power spectrum of the CARMA process.













   ..
       !! processed by numpydoc !!

.. py:function:: CARMA_autocovariance(tau, roots_AR, beta, sigma) -> jax.Array

   
   Computes the autocovariance of the CARMA process.


   :Parameters:

       **tau** : :obj:`float`
           Time lag at which the autocovariance is evaluated.

       **roots_AR** : :obj:`jax.Array`
           Roots of the AR polynomial.

       **beta** : :obj:`jax.Array`
           Coefficients of the MA polynomial.

       **sigma** : :obj:`float`
           Standard deviation of the white noise process.

   :Returns:

       :obj:`jax.Array`
           Autocovariance of the CARMA process.













   ..
       !! processed by numpydoc !!




