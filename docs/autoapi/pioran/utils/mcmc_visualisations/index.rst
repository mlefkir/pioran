
mcmc_visualisations
===================

.. py:module:: pioran.utils.mcmc_visualisations

.. autoapi-nested-parse::

   Visualisations/Diagnostics sampling.

   ..
       !! processed by numpydoc !!


Overview
--------


.. list-table:: Function
   :header-rows: 0
   :widths: auto
   :class: summarytable

   * - :py:obj:`from_samples_to_inference_data <pioran.utils.mcmc_visualisations.from_samples_to_inference_data>`\ (names, samples)
     - Convert the samples to an Arviz InferenceData object.
   * - :py:obj:`plot_diagnostics_sampling <pioran.utils.mcmc_visualisations.plot_diagnostics_sampling>`\ (dataset, plot_dir, prefix, plot_trace, plot_mcse, plot_rank, plot_ess)
     - Plot the diagnostics of the sampling.




Functions
---------
.. py:function:: from_samples_to_inference_data(names: List[str], samples)

   
   Convert the samples to an Arviz InferenceData object.


   :Parameters:

       **names: :obj:`List[str]`**
           The names of the parameters

       **samples: :obj:`numpy.ndarray`**
           The samples

   :Returns:

       :obj:`arviz.InferenceData`
           The InferenceData object













   ..
       !! processed by numpydoc !!

.. py:function:: plot_diagnostics_sampling(dataset, plot_dir, prefix='', plot_trace=True, plot_mcse=True, plot_rank=True, plot_ess=True)

   
   Plot the diagnostics of the sampling.


   :Parameters:

       **dataset: :obj:`arviz.InferenceData`**
           The InferenceData object

       **plot_dir: :obj:`str`**
           The directory to save the plots to

       **prefix: :obj:`str`, optional**
           The prefix to add to the plots. Default is ''

       **plot_trace: :obj:`bool`, optional**
           Whether to plot the trace plot. Default is True

       **plot_mcse: :obj:`bool`, optional**
           Whether to plot the MCSE plot. Default is True

       **plot_rank: :obj:`bool`, optional**
           Whether to plot the rank plot. Default is True

       **plot_ess: :obj:`bool`, optional**
           Whether to plot the ESS plot. Default is True














   ..
       !! processed by numpydoc !!




