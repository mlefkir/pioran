
inference_utils
===============

.. py:module:: pioran.utils.inference_utils


Overview
--------


.. list-table:: Function
   :header-rows: 0
   :widths: auto
   :class: summarytable

   * - :py:obj:`save_sampling_results <pioran.utils.inference_utils.save_sampling_results>`\ (info, warmup, samples, log_prob, log_densitygrad, filename)
     - Save the results of the Monte Carlo runs in a ASDF file.
   * - :py:obj:`progress_bar_factory <pioran.utils.inference_utils.progress_bar_factory>`\ (num_samples, num_chains)
     - Factory that builds a progress bar decorator along




Functions
---------
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




