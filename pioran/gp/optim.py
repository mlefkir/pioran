"""Class and function for the Gaussian Process Regression of 1D data.

"""

# modules for optimising the (hyper)parameters
import jax.numpy as jnp
from scipy.optimize import minimize
import ultranest

from .core import GaussianProcess


class Optimizer:
    """Class to optimize the (hyper)parameters of the Gaussian Process.
    
    Attributes
    ----------
    GP: GaussianProcess
        Gaussian Process object.
    initial_guess: list of float
        Initial guess for the (hyper)parameters.
    bounds: list of tuple
        Bounds for the (hyper)parameters.
    method: str
        Method to maximise the marginal likelihood.
        - "L-BFGS-B": using scipy.optimize.minimize
        - "nested": using nested sampling via ultranest
    
    results: dict
        Results of the optimization.
    
    Methods
    -------
    run:
        Optimize the (hyper)parameters of the Gaussian Process.
    Optimize_ML:
        Optimize the (hyper)parameters of the Gaussian Process using scipy.optimize.minimize.
    nested_sampling:
        Optimize the (hyper)parameters of the Gaussian Process using nested sampling via ultranest.
    
    """
    
    def __init__(self, GP: GaussianProcess, method="ML", x0=None,bounds=None):
        """Constructor method for the Optimizer class.

        Instantiate the Optimizer class.

        Parameters
        ----------
        GP: GaussianProcess
            Gaussian Process object.
        method: str, optional
            Method to maximise the marginal likelihood, by default "L-BFGS-B"
            - "ML: using scipy.optimize.minimize
            - "NS": using nested sampling via ultranest
        x0: list of floats, optional
            Initial guess for the (hyper)parameters, by default None (it will select the free parameters in GP.acvf.parameters.values)
        bounds: list of tuple, optional
            Bounds for the (hyper)parameters, by default None (it will select the free parameters in GP.acvf.parameters.boundaries)
            
        Raises
        ------
        TypeError
            If the method is not a string, if x0 is not a list of floats or if bounds is not a list of tuple.
        """
        
        self.GP = GP
        
        # set values for the (hyper)parameters to optimize
        if x0 is not None:
            # check if x0 is a list or a numpy array
            if isinstance(x0, list) or isinstance(x0, jnp.ndarray):
                self.initial_guess = x0 
                # add the initial guess of nu and mu if they are not in x0
                if  len(self.initial_guess) != len(self.GP.acvf.parameters.free_parameters):
                    if self.GP.estimate_mean: self.initial_guess.append(self.GP.acvf.parameters["nu"].value)
                    if self.GP.estimate_mean:  self.initial_guess.append(self.GP.acvf.parameters["mu"].value)   
                    assert len(self.initial_guess) == len(self.GP.acvf.parameters.free_parameters), f"The number of initial guesses ({len(self.initial_guess)}) is not the same as the number of free parameters ({len(self.GP.acvf.parameters.free_parameters)}) to optimize."
            else:
                raise TypeError("x0 must be a list or a numpy array.")
        # if x0 is None, use the values of the (hyper)parameters in GP.acvf.parameters.values
        else:
            self.initial_guess = [val for (val, free) in zip(self.GP.acvf.parameters.values, self.GP.acvf.parameters.free_parameters) if free]
        
        # set the boundaries
        if bounds is not None:
            if isinstance(bounds, list) or isinstance(bounds, jnp.ndarray):
                self.bounds = bounds
                if not len(self.bounds) == len(self.GP.acvf.parameters.free_parameters):
                    # add the boundaries of nu and mu
                    if self.GP.scale_errors: self.bounds.append(self.GP.acvf.parameters["nu"].bounds)
                    if self.GP.estimate_mean: self.bounds.append(self.GP.acvf.parameters["mu"].bounds)
                    assert len(self.bounds) == len(self.GP.acvf.parameters.free_parameters), f"The number of boundaries ({len(self.bounds)}) is not the same as the number of free parameters  ({len(self.GP.acvf.parameters.free_parameters)}) to optimize."
            else:
                raise TypeError("bounds must be a list or a numpy array.")
        else:
            self.bounds = self.GP.acvf.parameters.boundaries
        
        # set the method
        if isinstance(method, str):
            self.method = method
        else:
            raise TypeError("method must be a string.")   
         
    def run(self, priors=None, verbose=True, **kwargs):
        """ Optimize the (hyper)parameters of the Gaussian Process.
        
        
        Parameters
        ----------
        priors: function, optional, by default None
            Function to define the priors for the (hyper)parameters.
        verbose: bool, optional
            Print the results of the optimization, by default False
        **kwargs: dict
            Additional arguments for the optimization method.
                For ML: see 'optimize_ML' docstring
                For NS: see 'nested_sampling' docstring
        
        Raises
        ------
        ValueError
            If the method is not "ML" or "NS".
        
        Returns
        -------
        results: dict
            Results of the optimization. Different keys depending on the method.
        """
        if self.method == "ML":
            results = self.optimize_ML(verbose=verbose, **kwargs)
        elif self.method == "NS":
            if priors is None:
                raise ValueError("Priors must be provided for nested sampling.")
            results = self.nested_sampling(priors=priors,verbose=verbose,**kwargs)
        else:
            raise ValueError("The method must be either 'ML' or 'NS'.")
        print("\n>>>>>> Optimization done.")
        print(self.GP)
        return results


    def optimize_ML(self, verbose=True, **kwargs):
        """ Optimize the (hyper)parameters of the Gaussian Process with scipy.optimize.minimize.
        
        Parameters
        ----------
        verbose: bool, optional
            Print the results of the optimization, by default True
        **kwargs: dict
            Keyword arguments for scipy.optimize.minimize.
                gtol: float
                ftol: float
                maxiter: int
                
        Returns
        -------
        results: dict
            Results of the optimization. For details https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html#scipy.optimize.OptimizeResult
        """
        options = {'gtol': kwargs.get('gtol',1e-16),'ftol':kwargs.get('ftol',1e-16), 'disp': verbose}
        if 'maxiter' in kwargs:
            options['maxiter'] = kwargs['maxiter']
        method = kwargs.get('method', 'L-BFGS-B')
        
        res = minimize(self.GP.wrapper_neg_log_marginal_likelihood, self.initial_guess, method=method, bounds=self.bounds,options=options)
        self.GP.acvf.parameters.values =  res.x
        print("Optimization terminated successfully." if res.success else "/!\/!\/!\  Optimization failed! /!\/!\/!\ ")
        return res
        
    def nested_sampling(self,priors,verbose=True,**kwargs,):
        """ Optimize the (hyper)parameters of the Gaussian Process with nested sampling via ultranest.

        Parameters
        ----------
        priors: function
            Function to define the priors for the (hyper)parameters.
        verbose: bool, optional
            Print the results of the optimization and the progress of the sampling, by default True
        **kwargs: dict
            Keyword arguments for ultranest
                - resume: bool
                - log_dir: str
                - run_kwargs: dict
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
        
        sampler = ultranest.ReactiveNestedSampler(self.GP.acvf.parameters.names, self.GP.wrapper_log_marginal_likelihood,priors,resume=resume,log_dir=log_dir)
        results = sampler.run(**run_kwargs, **viz)
        sampler.plot()
        self.GP.acvf.parameters.values =  results['posterior']['median']
        return results 
    
def void(*args, **kwargs):
    """ Void function to avoid printing the status of the nested sampling."""
    pass