"""Class and functions for inference with Gaussian Processes and other methods.

"""
from typing import Union

import ultranest

from .core import GaussianProcess
from .carma_core import CARMAProcess

class Inference:
    r"""Class to estimate the (hyper)parameters of the Gaussian Process.
    
    Optimize the (hyper)parameters of the Gaussian Process using scipy.optimize.minimize or nested sampling via ultranest.
    
    Attributes
    ----------
    
    GP : :class:`~pioran.core.GaussianProcess`
        Gaussian Process object.
    initial_guess : list of float
        Initial guess for the (hyper)parameters.
    bounds : list of tuple
        Bounds for the (hyper)parameters.
    method : :obj:`str`
        - "L-BFGS-B": using scipy.optimize.minimize
        - "nested": using nested sampling via ultranest
    
    results : dict
        Results of the optimization.
    
    Methods
    -------
    
    run 
        Optimize the (hyper)parameters of the Gaussian Process.
    nested_sampling 
        Optimize the (hyper)parameters of the Gaussian Process using nested sampling via ultranest.
    
    """
    
    def __init__(self, Process: Union[GaussianProcess,CARMAProcess], method="NS"):
        r"""Constructor method for the Optimizer class.

        Instantiate the Inference class.

        Parameters
        ----------
        Process : :class:`~pioran.core.GaussianProcess`
            Gaussian Process object.
        method : str, optional
            "NS": using nested sampling via ultranest
    
        Raises
        ------
        TypeError
            If the method is not a string.
        """
        
        self.process = Process
        
        if isinstance(method, str):
            self.method = method
        else:
            raise TypeError("method must be a string.")   
        
         
    def run(self, priors=None, verbose=True, **kwargs):
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
            if priors is None:
                raise ValueError("Priors must be provided for nested sampling.")
            results = self.nested_sampling(priors=priors,verbose=verbose,**kwargs)
        else:
            raise ValueError("The method must be 'NS'.")
        print("\n>>>>>> Optimization done.")
        print(self.process)
        return results
        
    def nested_sampling(self,priors,verbose=True,**kwargs):
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
        sampler = ultranest.ReactiveNestedSampler(free_names, self.process.wrapper_log_marginal_likelihood,priors,resume=resume,log_dir=log_dir)
        if verbose: results = sampler.run(**viz)
        else: results = sampler.run(**run_kwargs, **viz)
        sampler.plot()
        self.process.model.parameters.set_free_values(results['posterior']['median'])
        return results 
    
def void(*args, **kwargs):
    """ Void function to avoid printing the status of the nested sampling."""
    pass