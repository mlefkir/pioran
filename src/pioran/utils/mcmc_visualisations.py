"""Visualisations/Diagnostics sampling."""

from typing import List

import arviz as az
import jax
import matplotlib.pyplot as plt
import numpy as np


def from_samples_to_inference_data(names:List[str],
                                   samples):
    """Convert the samples to an Arviz InferenceData object.
    
    Parameters
    ----------
    names: :obj:`List[str]`
        The names of the parameters
    samples: :obj:`numpy.ndarray`
        The samples
        
    Returns
    -------
    :obj:`arviz.InferenceData`
        The InferenceData object
    
    """
    if isinstance(samples, np.ndarray) or isinstance(samples, jax.Array):
        dic = {names[i]:samples[...,i,:] for i in range(len(names))}
    elif isinstance(samples, dict):
        num_chains = len(samples.keys())
        dic = {names[i]:np.array([samples[f'chain_{j}'] for j in range(num_chains)])[...,i,:] for i in range(len(names))}
    
    
    dataset = az.convert_to_inference_data(dic)
    return dataset


def plot_diagnostics_sampling(dataset,
                              plot_dir,
                              prefix='',
                              plot_trace=True,
                              plot_mcse=True,
                              plot_rank=True,
                              plot_ess=True):
    """Plot the diagnostics of the sampling.
    
    Parameters
    ----------
    dataset: :obj:`arviz.InferenceData`
        The InferenceData object
    plot_dir: :obj:`str`
        The directory to save the plots to
    prefix: :obj:`str`, optional
        The prefix to add to the plots. Default is ''
    plot_trace: :obj:`bool`, optional
        Whether to plot the trace plot. Default is True
    plot_mcse: :obj:`bool`, optional
        Whether to plot the MCSE plot. Default is True
    plot_rank: :obj:`bool`, optional
        Whether to plot the rank plot. Default is True
    plot_ess: :obj:`bool`, optional
        Whether to plot the ESS plot. Default is True    
    
    """
    
    
    if plot_trace:
        s = az.plot_trace(dataset)
        fig = plt.gcf()
        fig.suptitle('Trace plot')
        fig.tight_layout()
        fig.savefig(f'{plot_dir}/{prefix}trace_plot.pdf',bbox_inches='tight')
    
    if plot_rank:
        s = az.plot_rank(dataset)
        fig = plt.gcf()
        fig.suptitle('Rank plot (All chains)')
        fig.tight_layout()
        fig.savefig(f'{plot_dir}/{prefix}plot_rank.pdf',bbox_inches='tight')
    
    if plot_mcse:
        s = az.plot_mcse(dataset,extra_methods=True)
        fig = plt.gcf()
        fig.suptitle('Plot of the Monte Carlo Standard Error (MCSE) for each parameter')
        fig.tight_layout()
        fig.savefig(f'{plot_dir}/{prefix}mcse.pdf',bbox_inches='tight')
        
    if plot_ess:
        s = az.plot_ess(dataset,kind='quantile')
        fig = plt.gcf()
        fig.suptitle('Effective sample size for quantiles')
        fig.tight_layout()
        fig.savefig(f'{plot_dir}/{prefix}ess_quantiles.pdf')
        
        s = az.plot_ess(dataset,kind='local')
        fig = plt.gcf()
        fig.suptitle('Effective sample size for small intervals')
        fig.tight_layout()
        fig.savefig(f'{plot_dir}/{prefix}ess_small.pdf')