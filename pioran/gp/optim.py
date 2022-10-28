"""Class and function for the Gaussian Process Regression of 1D data.

"""

# modules for optimising the (hyper)parameters
import numpy as np
from scipy.optimize import minimize
import ultranest

from .core import GaussianProcess


class Optimizer:
    """Class to optimize the (hyper)parameters of the Gaussian Process.
    
    Attributes
    ----------
    GP : GaussianProcess
        Gaussian Process object.
    initial_guess : list of float
        Initial guess for the (hyper)parameters.
    bounds : list of tuple
        Bounds for the (hyper)parameters.
    method : str
        Method to maximise the marginal likelihood.
        - "L-BFGS-B" : using scipy.optimize.minimize
        - "nested" : using nested sampling via ultranest
    
    results : dict
        Results of the optimization.
    
    Methods
    -------
    
    """
    
    def __init__(self, GP: GaussianProcess, method="L-BFGS-B", x0=None,bounds=None):
        """Constructor method for the Optimizer class.

        Parameters
        ----------
        GP : GaussianProcess
            Gaussian Process object.
        method : str, optional
            Method to maximise the marginal likelihood, by default "L-BFGS-B"
            - "L-BFGS-B" : using scipy.optimize.minimize
            - "nested" : using nested sampling via ultranest
        x0 : list of floats, optional
            Initial guess for the (hyper)parameters, by default None (it will select the free parameters in GP.acvf.parameters.values)
        bounds : list of tuple, optional
            Bounds for the (hyper)parameters, by default None (it will select the free parameters in GP.acvf.parameters.boundaries)
        """
        
        self.GP = GP
        
        # set values for the (hyper)parameters to optimize
        if x0 is not None :
            # check if x0 is a list or a numpy array
            if isinstance(x0, list) or isinstance(x0, np.ndarray):
                self.initial_guess = x0 
                # add the initial guess of nu and mu if they are not in x0
                if  len(self.initial_guess) != len(self.GP.acvf.parameters.free_parameters):
                    if self.GP.estimate_mean : self.initial_guess.append(self.GP.acvf.parameters["nu"].value)
                    if self.GP.estimate_mean :  self.initial_guess.append(self.GP.acvf.parameters["mu"].value)   
                    assert len(self.initial_guess) == len(self.GP.acvf.parameters.free_parameters), f"The number of initial guesses ({len(self.initial_guess)}) is not the same as the number of free parameters ({len(self.GP.acvf.parameters.free_parameters)}) to optimize."
            else:
                raise TypeError("x0 must be a list or a numpy array.")
        # if x0 is None, use the values of the (hyper)parameters in GP.acvf.parameters.values
        else :
            self.initial_guess = [val for (val, free) in zip(self.GP.acvf.parameters.values, self.GP.acvf.parameters.free_parameters) if free]
        
        # set the boundaries
        if bounds is not None :
            if isinstance(bounds, list) or isinstance(bounds, np.ndarray):
                self.bounds = bounds
                if not len(self.bounds) == len(self.GP.acvf.parameters.free_parameters):
                    # add the boundaries of nu and mu
                    if self.GP.scale_errors : self.bounds.append(self.GP.acvf.parameters["nu"].bounds)
                    if self.GP.estimate_mean : self.bounds.append(self.GP.acvf.parameters["mu"].bounds)
                    assert len(self.bounds) == len(self.GP.acvf.parameters.free_parameters), f"The number of boundaries ({len(self.bounds)}) is not the same as the number of free parameters  ({len(self.GP.acvf.parameters.free_parameters)}) to optimize."
            else:
                raise TypeError("bounds must be a list or a numpy array.")
        else:
            self.bounds = self.GP.acvf.parameters.boundaries
        
        # set the method
        self.method = method
        

    def optimize_ML(self, verbose=False, **options):
        """ Optimize the (hyper)parameters of the Gaussian Process with scipy.optimize.minimize.
        """
        res = minimize(self.GP.wrapper_log_marginal_likelihood, self.initial_guess, method=self.method, bounds=self.bounds,options={'gtol': 1e-16,'ftol':1e-16, 'disp': True})
        self.GP.acvf.parameters.values =  res.x
        print(self.GP.acvf.parameters.values)
        print(res)
        print(self.GP)
        print("Optimization terminated successfully." if res.success else "/!\/!\/!\  Optimization failed! /!\/!\/!\ ")
        
    def nested_sampling(self,priors,resume=True,logdir=None):
        nested = ultranest.ReactiveNestedSampler(self.GP.acvf.parameters.names, self.GP.wrapper_neg_log_marginal_likelihood,priors,resume=resume,log_dir=logdir)
        results = nested.run()
        nested.plot()
        self.GP.acvf.parameters.values =  results['posterior']['median']
        return results 
    