
utils
=====

.. py:module:: pioran.utils

.. autoapi-nested-parse::

   
   Utility functions for pioran.
















   ..
       !! processed by numpydoc !!


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   ICCF/index.rst
   gp_utils/index.rst
   inference_utils/index.rst
   mcmc_visualisations/index.rst
   psd_utils/index.rst


Overview
--------


.. list-table:: Function
   :header-rows: 0
   :widths: auto
   :class: summarytable

   * - :py:obj:`EuclideanDistance <pioran.utils.EuclideanDistance>`\ (xq, xp)
     - Compute the Euclidean distance between two arrays.
   * - :py:obj:`nearest_positive_definite <pioran.utils.nearest_positive_definite>`\ (A)
     - Find the nearest positive-definite matrix to input.
   * - :py:obj:`progress_bar_factory <pioran.utils.progress_bar_factory>`\ (num_samples, num_chains)
     - Factory that builds a progress bar decorator along
   * - :py:obj:`save_sampling_results <pioran.utils.save_sampling_results>`\ (info, warmup, samples, log_prob, log_densitygrad, filename)
     - Save the results of the Monte Carlo runs in a ASDF file.
   * - :py:obj:`SHO_power_spectrum <pioran.utils.SHO_power_spectrum>`\ (f, A, f0)
     - Power spectrum of a stochastic harmonic oscillator.
   * - :py:obj:`get_samples_psd <pioran.utils.get_samples_psd>`\ (psd_acvf, f, params_samples)
     - Just a wrapper for jax.vmap(get_psd_approx_samples,(None,None,0))(psd_acvf,f,params_samples)
   * - :py:obj:`wrapper_psd_true_samples <pioran.utils.wrapper_psd_true_samples>`\ (psd_acvf, f, params_samples)
     - Just a wrapper for jax.vmap(get_psd_true_samples,(None,None,0))(psd_acvf,f,params_samples)




Functions
---------
.. py:function:: EuclideanDistance(xq, xp)

   
   Compute the Euclidean distance between two arrays.

   .. math:: :label: euclidian_distance

       D(\boldsymbol{x_q},\boldsymbol{x_p}) = \sqrt{(\boldsymbol{x_q} - \boldsymbol{x_p}^{\mathrm{T}})^2}

   :Parameters:

       **xq** : (n, 1) :obj:`jax.Array`
           First array.

       **xp** : (m, 1) :obj:`jax.Array`
           Second array.

       **Returns**
           ..

       **-------**
           ..

       **(n, m) :obj:`jax.Array`**
           ..














   ..
       !! processed by numpydoc !!

.. py:function:: nearest_positive_definite(A)

   
   Find the nearest positive-definite matrix to input.

   Code from Ahmed Fasih - https://gist.github.com/fasiha/fdb5cec2054e6f1c6ae35476045a0bbd
   A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
   credits [2].

   :Parameters:

       **A** : (N, N) :obj:`jax.Array`
           Matrix to find the nearest positive-definite

   :Returns:

       (N, N) :obj:`jax.Array`
           Nearest positive-definite matrix to A.








   .. rubric:: Notes

   1. https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
   2. N.J. Higham, "Computing a nearest symmetric positive semidefinite" (1988): https://doi.org/10.1016/0024-3795(88)90223-6





   ..
       !! processed by numpydoc !!

.. py:function:: progress_bar_factory(num_samples: int, num_chains: int)

   
   Factory that builds a progress bar decorator along
   with the `set_tqdm_description` and `close_tqdm` functions

   progress bar obtained from numpyro source code
   https://github.com/pyro-ppl/numpyro/blob/6a0856b7cda82fc255e23adc797bb79f5b7fc904/numpyro/util.py#L176
   and modified to work with scan using https://www.jeremiecoullon.com/2021/01/29/jax_progress_bar/

   :Parameters:

       **num_samples: :obj:`int`**
           The number of samples

       **num_chains: :obj:`int`**
           The number of chains














   ..
       !! processed by numpydoc !!

.. py:function:: save_sampling_results(info: dict, warmup: dict, samples: numpy.ndarray, log_prob: numpy.ndarray, log_densitygrad: numpy.ndarray, filename: str)

   
   Save the results of the Monte Carlo runs in a ASDF file.

   This file contains the following data:
   - info: a dictionary containing the information about the run, namely:
       - num_params: the number of parameters
       - num_samples: the number of samples
       - num_warmup: the number of warmup samples
       - num_chains: the number of chains
       - ESS: the effective sample size
       - Rhat-split: the split Rhat statistic
   - warmup: a numpy array containing the warmup samples
   - samples: a numpy array containing the samples
   - log_prob: a numpy array containing the log probabilities of the samples

   :Parameters:

       **info: :obj:`dict`**
           A dictionary containing the information about the run

       **warmup: :obj:`dict`**
           A numpy array containing the warmup samples

       **samples: :obj:`jax.Array`**
           A numpy array containing the samples

       **log_prob: :obj:`jax.Array`**
           A numpy array containing the log probabilities of the samples

       **log_densitygrad: :obj:`jax.Array`**
           A numpy array containing the log density gradients of the samples

       **filename: :obj:`str`**
           The name of the file to save the data to














   ..
       !! processed by numpydoc !!

.. py:function:: SHO_power_spectrum(f: jax.Array, A: float, f0: float) -> jax.Array

   
   Power spectrum of a stochastic harmonic oscillator.

   .. math:: :label: sho_power_spectrum

      \mathcal{P}(f) = \dfrac{A}{1 + (f-f_0)^4}.

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




