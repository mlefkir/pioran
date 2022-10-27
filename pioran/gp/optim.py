"""Class and function for the Gaussian Process Regression of 1D data.

"""

# modules for optimising the (hyper)parameters
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
            Initial guess for the (hyper)parameters, by default None (it will read GP.acf.parameters.values)
        bounds : list of tuple, optional
            Bounds for the (hyper)parameters, by default None (it will read GP.acf.parameters.boundaries)
        """
        
        self.GP = GP
        
        # set values for the (hyper)parameters to optimize
        if x0 is not None :
            self.initial_guess = x0 
            # add the initial guess of nu and mu if they are not in x0
            if  len(self.initial_guess) != len(self.GP.acf.parameters.free_parameters):
                if self.GP.estimate_mean : self.initial_guess.append(self.GP.acf.parameters["nu"])
                if self.GP.estimate_mean :  self.initial_guess.append(self.GP.acf.parameters["mu"])   
                assert len(self.initial_guess) == len(self.GP.acf.parameters.free_parameters), "The number of initial guess is not the same as the number of parameters to optimize."
        else :
            self.initial_guess = self.GP.acf.parameters.values
        
        # set the boundaries
        if bounds is not None :
            self.bounds = bounds
            if not len(self.initial_guess) == len(self.GP.acf.parameters.values):
                # add the boundaries of nu and mu
                if self.GP.scale_errors : self.bounds.append(self.GP.acf.parameters.boundaries["nu"])
                if self.GP.estimate_mean : self.bounds.append(self.GP.acf.parameters.boundaries["mu"])
        else:
            self.bounds = [self.GP.acf.parameters.boundaries[name] for name in self.GP.acf.parameters.names ]
        
        assert len(self.bounds) == len(self.initial_guess), "The number of boundaries is not the same as the number of parameters to optimize."
        self.method = method
        

    def optimize_ML(self):
        res = minimize(self.GP.wrapperLogMarginalLikelihood, self.initial_guess, method=self.method, bounds=self.bounds,options={'gtol': 1e-16,'ftol':1e-16, 'disp': True})
        self.GP.acf.parameters.values =  res.x
        print(self.GP.acf.parameters.values)
        print(res)
        self.GP.acf.print_info()
        print("Optimization terminated successfully." if res.success else "/!\/!\/!\  Optimization failed! /!\/!\/!\ ")
        
    def nested_sampling(self,priors,resume=True,logdir=None):
        nested = ultranest.ReactiveNestedSampler(self.GP.acf.parameters.names, self.GP.wrapperNegLogMarginalLikelihood,priors,resume=resume,log_dir=logdir)
        results = nested.run()
        nested.plot()
        self.GP.acf.parameters.values =  results['posterior']['median']
        return results 
    