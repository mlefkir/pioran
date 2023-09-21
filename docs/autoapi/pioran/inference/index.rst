
inference
=========

.. py:module:: pioran.inference

.. autoapi-nested-parse::

   Infer the (hyper)parameters of processes.

   ..
       !! processed by numpydoc !!


Overview
--------

.. list-table:: Classes
   :header-rows: 0
   :widths: auto
   :class: summarytable

   * - :py:obj:`Inference <pioran.inference.Inference>`
     - Class to infer the value of the (hyper)parameters of the Gaussian Process.


.. list-table:: Function
   :header-rows: 0
   :widths: auto
   :class: summarytable

   * - :py:obj:`void <pioran.inference.void>`\ (\*args, \*\*kwargs)
     - Void function to avoid printing the status of the nested sampling.


.. list-table:: Attributes
   :header-rows: 0
   :widths: auto
   :class: summarytable

   * - :py:obj:`inference_methods <pioran.inference.inference_methods>`
     - List of inference methods implemented.
   * - :py:obj:`blackjax <pioran.inference.blackjax>`
     - \-
   * - :py:obj:`ultranest <pioran.inference.ultranest>`
     - \-
   * - :py:obj:`comm <pioran.inference.comm>`
     - \-


Classes
-------

.. py:class:: Inference(Process: pioran.core.GaussianProcess | pioran.carma.carma_core.CARMAProcess, priors, method, n_samples_checks=1000, seed_check=0, run_checks=True, log_dir='log_dir')

   
   Class to infer the value of the (hyper)parameters of the Gaussian Process.

   Various methods to sample the posterior probability distribution of the (hyper)parameters of the Gaussian Process are implemented
   as wrappers around the inference packages `blackjax` and `ultranest`.

   :Parameters:

       **Process** : :class:`~pioran.core.GaussianProcess`
           Process object.

       **priors** : :obj:`function`
           Function to define the priors for the (hyper)parameters.

       **method** : :obj:`str`, optional
           "NS": using nested sampling via ultranest

       **n_samples_checks** : :obj:`int`, optional
           Number of samples to take from the prior distribution, by default 1000

       **seed_check** : :obj:`int`, optional
           Seed for the random number generator, by default 0

       **run_checks** : :obj:`bool`, optional
           Run the prior predictive checks, by default True

       **log_dir** : :obj:`str`, optional
           Directory to save the results of the inference, by default 'log_dir'





   :Raises:

       ImportError
           If the required packages are not installed.

       ValueError
           If the saved config file is different from the current config, or if the method is not valid.

       TypeError
           If the method is not a string.









   ..
       !! processed by numpydoc !!

   .. rubric:: Overview

   .. list-table:: Attributes
      :header-rows: 0
      :widths: auto
      :class: summarytable

      * - :py:obj:`process <pioran.inference.Inference.process>`
        - Process object.
      * - :py:obj:`n_pars <pioran.inference.Inference.n_pars>`
        - Number of (hyper)parameters.
      * - :py:obj:`priors <pioran.inference.Inference.priors>`
        - Function to define the priors for the (hyper)parameters.
      * - :py:obj:`log_dir <pioran.inference.Inference.log_dir>`
        - Directory to save the results of the inference.
      * - :py:obj:`plot_dir <pioran.inference.Inference.plot_dir>`
        - Directory to save the plots of the inference.
      * - :py:obj:`method <pioran.inference.Inference.method>`
        - Method to use for the inference.


   .. list-table:: Methods
      :header-rows: 0
      :widths: auto
      :class: summarytable

      * - :py:obj:`save_config <pioran.inference.Inference.save_config>`\ (save_file)
        - Save the configuration of the inference.
      * - :py:obj:`prior_predictive_checks <pioran.inference.Inference.prior_predictive_checks>`\ (n_samples_checks, seed_check, n_frequencies, plot_prior_samples, plot_prior_predictive_distribution)
        - Check the prior predictive distribution.
      * - :py:obj:`check_approximation <pioran.inference.Inference.check_approximation>`\ (n_samples_checks, seed_check, n_frequencies, plot_diagnostics, plot_violins, plot_quantiles)
        - Check the approximation of the PSD with the kernel decomposition.
      * - :py:obj:`run <pioran.inference.Inference.run>`\ (verbose, user_log_likelihood, seed, n_chains, n_samples, n_warmup_steps, use_stepsampler)
        - Estimate the (hyper)parameters of the Gaussian Process.
      * - :py:obj:`blackjax_NUTS <pioran.inference.Inference.blackjax_NUTS>`\ (rng_key, initial_position, log_likelihood, log_prior, num_warmup_steps, num_samples, num_chains)
        - Sample the posterior distribution using the NUTS sampler from blackjax.
      * - :py:obj:`nested_sampling <pioran.inference.Inference.nested_sampling>`\ (priors, log_likelihood, verbose, use_stepsampler, resume, run_kwargs, slice_steps)
        - Sample the posterior distribution of the (hyper)parameters of the Gaussian Process with nested sampling via ultranest.


   .. rubric:: Members

   .. py:attribute:: process
      :type: pioran.core.GaussianProcess | pioran.carma.carma_core.CARMAProcess

      
      Process object.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: n_pars
      :type: int

      
      Number of (hyper)parameters.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: priors
      :type: callable

      
      Function to define the priors for the (hyper)parameters.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: log_dir
      :type: str

      
      Directory to save the results of the inference.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: plot_dir
      :type: str

      
      Directory to save the plots of the inference.
















      ..
          !! processed by numpydoc !!

   .. py:attribute:: method
      :type: str

      
      Method to use for the inference.
















      ..
          !! processed by numpydoc !!

   .. py:method:: save_config(save_file=True)

      
      Save the configuration of the inference.

      Save the configuration of the inference, process and model in a json file.

      :Parameters:

          **save_file** : :obj:`bool`, optional
              ..

      :Returns:

          **dict_config** : :obj:`dict`
              Dictionary with the configuration of the inference, process and model.













      ..
          !! processed by numpydoc !!

   .. py:method:: prior_predictive_checks(n_samples_checks, seed_check, n_frequencies=1000, plot_prior_samples=True, plot_prior_predictive_distribution=True)

      
      Check the prior predictive distribution.

      Get samples from the prior distribution and plot them, and calculate the prior predictive
      distribution of the model and plot it.

      :Parameters:

          **n_samples_checks** : :obj:`int`
              Number of samples to take from the prior distribution, by default 1000

          **seed_check** : :obj:`int`
              Seed for the random number generator

          **plot_prior_samples** : :obj:`bool`, optional
              Plot the prior samples, by default True

          **plot_prior_predictive_distributions** : :obj:`bool`, optional
              Plot the prior predictive distribution of the model, by default True














      ..
          !! processed by numpydoc !!

   .. py:method:: check_approximation(n_samples_checks: int, seed_check: int, n_frequencies: int = 1000, plot_diagnostics: bool = True, plot_violins: bool = True, plot_quantiles: bool = True)

      
      Check the approximation of the PSD with the kernel decomposition.

      This method will take random samples from the prior distribution and compare the PSD obtained
      with the SHO decomposition with the true PSD.

      :Parameters:

          **n_samples_checks** : :obj:`int`
              Number of samples to take from the prior distribution, by default 1000

          **seed_check** : :obj:`int`
              Seed for the random number generator

          **n_frequencies** : :obj:`int`, optional
              Number of frequencies to evaluate the PSD, by default 1000

          **plot_diagnostics** : :obj:`bool`, optional
              Plot the diagnostics of the approximation, by default True

          **plot_violins** : :obj:`bool`, optional
              Plot the violin plots of the residuals and the ratios, by default True

          **plot_quantiles** : :obj:`bool`, optional
              Plot the quantiles of the residuals and the ratios, by default True

          **plot_prior_samples** : :obj:`bool`, optional
              Plot the prior samples, by default True

      :Returns:

          **figs** : :obj:`list`
              List of figures.

          **residuals** : :obj:`jax.Array`
              Residuals of the PSD approximation.

          **ratio** : :obj:`jax.Array`
              Ratio of the PSD approximation.













      ..
          !! processed by numpydoc !!

   .. py:method:: run(verbose: bool = True, user_log_likelihood=None, seed: int = 0, n_chains: int = 1, n_samples: int = 1000, n_warmup_steps: int = 1000, use_stepsampler: bool = False)

      
      Estimate the (hyper)parameters of the Gaussian Process.

      Run the inference method.

      :Parameters:

          **verbose** : :obj:`bool`, optional
              Be verbose, by default True

          **user_log_likelihood** : :obj:`function`, optional
              User-defined function to compute the log-likelihood, by default None

          **seed** : :obj:`int`, optional
              Seed for the random number generator, by default 0

          **n_chains** : :obj:`int`, optional
              Number of chains, by default 1

          **n_samples** : :obj:`int`, optional
              Number of samples to take from the posterior distribution, by default 1_000

          **n_warmup_steps** : :obj:`int`, optional
              Number of warmup steps, by default 1_000

          **use_stepsampler** : :obj:`bool`, optional
              Use the slice sampler as step sampler, by default False

      :Returns:

          results: dict
              Results of the sampling. The keys differ depending on the method/sampler used.













      ..
          !! processed by numpydoc !!

   .. py:method:: blackjax_NUTS(rng_key: jax.random.PRNGKey, initial_position: jax.Array, log_likelihood: callable, log_prior: callable, num_warmup_steps: int = 1000, num_samples: int = 1000, num_chains: int = 1)

      
      Sample the posterior distribution using the NUTS sampler from blackjax.

      Wrapper around the NUTS sampler from blackjax to sample the posterior distribution.
      This function also performs the warmup via window adaptation.

      :Parameters:

          **rng_key** : :obj:`jax.random.PRNGKey`
              Random key for the random number generator.

          **initial_position** : :obj:`jax.Array`
              Initial position of the chains.

          **log_likelihood** : :obj:`function`
              Function to compute the log-likelihood.

          **log_prior** : :obj:`function`
              Function to compute the log-prior.

          **num_warmup_steps** : :obj:`int`, optional
              Number of warmup steps, by default 1_000

          **num_samples** : :obj:`int`, optional
              Number of samples to take from the posterior distribution, by default 1_000

          **num_chains** : :obj:`int`, optional
              Number of chains, by default 1

      :Returns:

          **samples** : :obj:`jax.Array`
              Samples from the posterior distribution. It has shape (num_chains, num_params, num_samples).

          **log_prob** : :obj:`jax.Array`
              Log-probability of the samples.













      ..
          !! processed by numpydoc !!

   .. py:method:: nested_sampling(priors: callable, log_likelihood: callable, verbose: bool = True, use_stepsampler: bool = False, resume: bool = True, run_kwargs={}, slice_steps=100)

      
      Sample the posterior distribution of the (hyper)parameters of the Gaussian Process with nested sampling via ultranest.

      Perform nested sampling to sample the (hyper)parameters of the Gaussian Process.

      :Parameters:

          **priors** : :obj:`function`
              Function to define the priors for the parameters

          **log_likelihood** : :obj:`function`
              Function to compute the log-likelihood.

          **verbose** : :obj:`bool`, optional
              Print the results of the sample and the progress of the sampling, by default True

          **use_stepsampler** : :obj:`bool`, optional
              Use the slice sampler as step sampler, by default False

          **resume** : :obj:`bool`, optional
              Resume the sampling from the previous run, by default True

          **run_kwargs** : :obj:`dict`, optional
              Dictionary of arguments for ReactiveNestedSampler.run() see https://johannesbuchner.github.io/UltraNest/ultranest.html#module-ultranest.integrator

          **slice_steps** : :obj:`int`, optional
              Number of steps for the slice sampler, by default 100

      :Returns:

          results: dict
              Dictionary of results from the nested sampling.













      ..
          !! processed by numpydoc !!



Functions
---------
.. py:function:: void(*args, **kwargs)

   
   Void function to avoid printing the status of the nested sampling.
















   ..
       !! processed by numpydoc !!


Attributes
----------
.. py:data:: inference_methods
   :value: ['ultranest', 'blackjax_nuts']

   
   List of inference methods implemented.
















   ..
       !! processed by numpydoc !!

.. py:data:: blackjax

   

.. py:data:: ultranest

   

.. py:data:: comm

   



