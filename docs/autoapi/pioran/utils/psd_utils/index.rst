
psd_utils
=========

.. py:module:: pioran.utils.psd_utils

.. autoapi-nested-parse::

   Utility functions for PSD models.

   ..
       !! processed by numpydoc !!


Overview
--------


.. list-table:: Function
   :header-rows: 0
   :widths: auto
   :class: summarytable

   * - :py:obj:`SHO_power_spectrum <pioran.utils.psd_utils.SHO_power_spectrum>`\ (f, A, f0)
     - Power spectrum of a stochastic harmonic oscillator.
   * - :py:obj:`DRWCelerite_power_spectrum <pioran.utils.psd_utils.DRWCelerite_power_spectrum>`\ (f, A, f0)
     - Power spectrum of the DRW+Celerite component.
   * - :py:obj:`SHO_autocovariance <pioran.utils.psd_utils.SHO_autocovariance>`\ (tau, A, f0)
     - Autocovariance function of a stochastic harmonic oscillator.
   * - :py:obj:`get_psd_approx_samples <pioran.utils.psd_utils.get_psd_approx_samples>`\ (psd_acvf, f, params_samples)
     - Get the true PSD model and the approximated PSD using SHO decomposition.
   * - :py:obj:`get_samples_psd <pioran.utils.psd_utils.get_samples_psd>`\ (psd_acvf, f, params_samples)
     - Just a wrapper for jax.vmap(get_psd_approx_samples,(None,None,0))(psd_acvf,f,params_samples)
   * - :py:obj:`get_psd_true_samples <pioran.utils.psd_utils.get_psd_true_samples>`\ (psd_acvf, f, params_samples)
     - Get the true PSD model.
   * - :py:obj:`wrapper_psd_true_samples <pioran.utils.psd_utils.wrapper_psd_true_samples>`\ (psd_acvf, f, params_samples)
     - Just a wrapper for jax.vmap(get_psd_true_samples,(None,None,0))(psd_acvf,f,params_samples)




Functions
---------
.. py:function:: SHO_power_spectrum(f: jax.Array, A: float, f0: float) -> jax.Array

   
   Power spectrum of a stochastic harmonic oscillator.

   .. math:: :label: sho_power_spectrum

      \mathcal{P}(f) = \dfrac{A}{1 + (f/f_0)^4}.

   with the amplitude :math:`A`, the position :math:`f_0\ge 0`.

   :Parameters:

       **f** : :obj:`jax.Array`
           Frequency array.

       **A** : :obj:`float`
           Amplitude.

       **f0** : :obj:`float`
           Position.

   :Returns:

       :obj:`jax.Array`
           ..













   ..
       !! processed by numpydoc !!

.. py:function:: DRWCelerite_power_spectrum(f: jax.Array, A: float, f0: float) -> jax.Array

   
   Power spectrum of the DRW+Celerite component.

   .. math:: :label: drwcel_power_spectrum

      \mathcal{P}(f) = \dfrac{A}{1 + (f/f_0)^6}.

   with the amplitude :math:`A`, the position :math:`f_0\ge 0`.

   :Parameters:

       **f** : :obj:`jax.Array`
           Frequency array.

       **A** : :obj:`float`
           Amplitude.

       **f0** : :obj:`float`
           Position.

   :Returns:

       :obj:`jax.Array`
           ..













   ..
       !! processed by numpydoc !!

.. py:function:: SHO_autocovariance(tau: jax.Array, A: float, f0: float) -> jax.Array

   
   Autocovariance function of a stochastic harmonic oscillator.

   .. math:: :label: sho_autocovariance

      K(\tau) = A \times 2\pi f_0 \exp\left(-\dfrac{ 2\pi f_0 \tau}{\sqrt{2}}\right) \cos\left(\dfrac{ 2\pi f_0 \tau}{\sqrt{2}}-\dfrac{\pi}{4}\right).

   with the amplitude :math:`A`, the position :math:`f_0\ge 0`.

   :Parameters:

       **tau** : :obj:`jax.Array`
           Time lag array.

       **A** : :obj:`float`
           Amplitude.

       **f0** : :obj:`float`
           Position.

   :Returns:

       :obj:`jax.Array`
           ..













   ..
       !! processed by numpydoc !!

.. py:function:: get_psd_approx_samples(psd_acvf: pioran.psdtoacv.PSDToACV, f: jax.Array, params_samples: jax.Array) -> jax.Array

   
   Get the true PSD model and the approximated PSD using SHO decomposition.

   Given a PSDToACV object and a set of parameters, return the true PSD and the approximated PSD using SHO decomposition.

   :Parameters:

       **psd_acvf** : :class:`~pioran.psdtoacv.PSDToACV`
           PSDToACV object.

       **f** : :obj:`jax.Array`
           Frequency array.

       **params_samples** : :obj:`jax.Array`
           Parameters of the PSD model.

   :Returns:

       :obj:`jax.Array`
           True PSD.

       :obj:`jax.Array`
           Approximated PSD.













   ..
       !! processed by numpydoc !!

.. py:function:: get_samples_psd(psd_acvf: pioran.psdtoacv.PSDToACV, f: jax.Array, params_samples: jax.Array) -> jax.Array

   
   Just a wrapper for jax.vmap(get_psd_approx_samples,(None,None,0))(psd_acvf,f,params_samples)


   :Parameters:

       **psd_acvf** : :class:`~pioran.psdtoacv.PSDToACV`
           PSDToACV object.

       **f** : :obj:`jax.Array`
           Frequency array.

       **params_samples** : :obj:`jax.Array`
           Parameters of the PSD model.














   ..
       !! processed by numpydoc !!

.. py:function:: get_psd_true_samples(psd_acvf: pioran.psdtoacv.PSDToACV, f: jax.Array, params_samples: jax.Array) -> jax.Array

   
   Get the true PSD model.


   :Parameters:

       **psd_acvf** : :class:`~pioran.psdtoacv.PSDToACV`
           PSDToACV object.

       **f** : :obj:`jax.Array`
           Frequency array.

       **params_samples** : :obj:`jax.Array`
           Parameters of the PSD model.














   ..
       !! processed by numpydoc !!

.. py:function:: wrapper_psd_true_samples(psd_acvf: pioran.psdtoacv.PSDToACV, f: jax.Array, params_samples: jax.Array) -> jax.Array

   
   Just a wrapper for jax.vmap(get_psd_true_samples,(None,None,0))(psd_acvf,f,params_samples)


   :Parameters:

       **psd_acvf** : :class:`~pioran.psdtoacv.PSDToACV`
           PSDToACV object.

       **f** : :obj:`jax.Array`
           Frequency array.

       **params_samples** : :obj:`jax.Array`
           Parameters of the PSD model.














   ..
       !! processed by numpydoc !!




