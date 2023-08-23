"""Class and functions for inference with Gaussian Processes and other methods.

"""
from typing import Union

import jax
import jax.numpy as jnp
import numpy as np
import ultranest
import ultranest.stepsampler
from mpi4py import MPI

from .carma_core import CARMAProcess
from .core import GaussianProcess
from .psdtoacv import PSDToACV
from .utils.psd_decomp import get_samples_psd
from .plots import violin_plots_psd_approx,diagnostics_psd_approx,residuals_quantiles
from .utils.gp_utils import tinygp_methods


comm = MPI.COMM_WORLD
rank = comm.Get_rank()

class Inference:
    r"""Class to infer the value of the (hyper)parameters of the Gaussian Process.
    
    
    Attributes
    ----------
    
    process : :class:`~pioran.core.GaussianProcess`
        Gaussian Process object.
    priors: :obj:`function`
        Function to define the priors for the (hyper)parameters.
    method : :obj:`str`
        - "ultranest": nested sampling via ultranest.
    results : :obj:`dict`
        Results of the inference.
    
    Methods
    -------
    
    run 
        Optimize the (hyper)parameters of the Gaussian Process.
    nested_sampling 
        Optimize the (hyper)parameters of the Gaussian Process using nested sampling via ultranest.
    
    """
    
    def __init__(self, Process: Union[GaussianProcess,CARMAProcess],priors, method :str="ultranest",n_samples=10000,seed_check=0):
        r"""Constructor method for the Optimizer class.

        Instantiate the Inference class.

        Parameters
        ----------
        Process : :class:`~pioran.core.GaussianProcess`
            Process object.
        priors : :obj:`function`
            Function to define the priors for the (hyper)parameters.
        method : str, optional
            "NS": using nested sampling via ultranest
        n_samples : int, optional
            Number of samples to take from the prior distribution, by default 1000
        seed_check : int, optional
            Seed for the random number generator, by default 0    
            
        Raises
        ------
        TypeError
            If the method is not a string.
        """
        
        self.process = Process
        self.priors = priors
        
        
        if isinstance(method, str):
            self.method = method
        else:
            raise TypeError("method must be a string.")  
        
        if isinstance(Process, GaussianProcess) and isinstance(self.process.model,PSDToACV):
            if self.process.model.method in tinygp_methods:
                print(f"The PSD model is a {self.process.model.method} decomposition, checking the approximation.")
                self.check_approximation(n_samples,seed_check)
        
    def check_approximation(self,n_samples,seed_check,n_frequencies=1000):
        """Check the approximation of the PSD with the kernel decomposition.
        
        This method will take random samples from the prior distribution and compare the PSD obtained 
        with the SHO decomposition with the true PSD.

        Parameters
        ----------
        n_samples : int, optional
            Number of samples to take from the prior distribution, by default 1000
        seed : int, optional
            , by default 0
        """
        freqs = jnp.geomspace(self.process.model.f0, self.process.model.fN, n_frequencies)
        n_pars = len(self.process.model.parameters)

        if self.method == 'ultranest':
            # draw samples from the prior distribution
            rng = np.random.default_rng(seed=seed_check)  
            uniform_samples = rng.uniform(size=(n_pars,n_samples))  
            params_samples = self.priors(uniform_samples)#[indexes]

        else:
            raise NotImplementedError("Only ultranest is implemented for now.")   
        
        
        # get the true PSD and the SHO PSD samples        
        psd_true,psd_approx = get_samples_psd(self.process.model,freqs,params_samples.T)
        # compute the residuals and the ratios
        residuals = psd_true-psd_approx
        ratio = psd_approx/psd_true
        fig,_ = diagnostics_psd_approx(f=freqs,
                                        res=residuals,
                                        ratio=ratio,
                                        f_min=self.process.model.f_min_obs,
                                        f_max=self.process.model.f_max_obs)
        fig.savefig("diagnostics_psd_approx.pdf")
        
        fig2,_ = violin_plots_psd_approx(res=residuals,
                                        ratio=ratio)
        fig2.savefig("violin_plots_psd_approx.pdf")
        
        fig3,_ = residuals_quantiles(residuals=residuals,
                                     ratio=ratio,
                                     f=freqs,
                                     f_min=self.process.model.f_min_obs,
                                     f_max=self.process.model.f_max_obs)
        return fig,fig2,fig3,residuals,ratio
                    
         
    def run(self, verbose=True, **kwargs):
        """ Optimize the (hyper)parameters of the Gaussian Process.
        
        Run the inference method.
        
        Parameters
        ----------
        priors : :obj:`function`, optional
            Function to define the priors for the (hyper)parameters.
        verbose : :obj:`bool`, optional
            Print the results of the optimization, by default False
        **kwargs : :obj:`dict`, optional
            Additional arguments for the optimization method.
                For ML: see 'optimize_ML' docstring
        
        Raises
        ------
        ValueError
            If the method is not "NS".
        
        Returns
        -------
        results: dict
            Results of the optimization. Different keys depending on the method.
        """
        if self.method == "NS":
            use_stepsampler = kwargs.pop('use_stepsampler',False)
            if 'user_likelihood' in kwargs:
                print("user_likelihood is used please check the documentation.")
            user_likelihood = kwargs.pop('user_likelihood',self.process.wrapper_log_marginal_likelihood)
            results, sampler = self.nested_sampling(priors=self.priors,user_likelihood=user_likelihood,verbose=verbose,use_stepsampler=use_stepsampler,**kwargs)
        else:
            raise ValueError("The method must be 'NS'.")
        comm.Barrier()
        self.process.model.parameters.set_free_values(results['maximum_likelihood']['point'])#results['posterior']['median'])
        print(self.process.model.parameters.free_values)
        if rank == 0:
            print("\n>>>>>> Plotting corner and trace.")
            sampler.plot()
            print("\n>>>>>> Optimization done.")
            print(self.process)
        return results
        
    def nested_sampling(self,priors,user_likelihood,verbose=True,use_stepsampler=False,**kwargs):
        r""" Optimize the (hyper)parameters of the Gaussian Process with nested sampling via ultranest.

        Perform nested sampling to optimize the (hyper)parameters of the Gaussian Process.    

        Parameters
        ----------
        priors : :obj:`function`
            Function to define the priors for the parameters to be optimized.
        verbose : :obj:`bool`, optional
            Print the results of the optimization and the progress of the sampling, by default True
        **kwargs : :obj:`dict`
            Keyword arguments for ultranest
                - resume: :obj:`bool`
                - log_dir: :obj:`str`
                - run_kwargs: :obj:`dict`
                - Dictionary of arguments for ReactiveNestedSampler.run() see https://johannesbuchner.github.io/UltraNest/ultranest.html#module-ultranest.integrator
        
        Returns
        -------
        results: dict
            Dictionary of results from the nested sampling. 
        """
        
        resume = kwargs.get('resume',True)
        log_dir = kwargs.get('log_dir','GP_ultranest')
        run_kwargs = kwargs.get('run_kwargs',{})
        viz = {} if verbose else  {'show_status': False , 'viz_callback': void}
        free_names = self.process.model.parameters.free_names
        slice_steps = kwargs.get('slice_steps',100)
        sampler = ultranest.ReactiveNestedSampler(free_names,user_likelihood ,priors,resume=resume,log_dir=log_dir)
        if use_stepsampler: sampler.stepsampler = ultranest.stepsampler.SliceSampler(nsteps=slice_steps,
                                                generate_direction=ultranest.stepsampler.generate_mixture_random_direction)
        
        if verbose: results = sampler.run(**viz)
        else: results = sampler.run(**run_kwargs, **viz)
        
        return results,sampler
    
def void(*args, **kwargs):
    """ Void function to avoid printing the status of the nested sampling."""
    pass