# modules for optimising the hyperparameters
from scipy.optimize import minimize
import ultranest
from .GP_core import GaussianProcess


class Optimizer:
    
    def __init__(self, GP: GaussianProcess, method="L-BFGS-B", x0=None,bounds=None):
        self.GP = GP
        self.initial_guess = x0
        self.bounds = bounds
        self.method = method
        
    def optimize_ML(self):
        optimizer = minimize(self.GP.wrapperLogMarginalLikelihood, self.initial_guess, method=self.method, bounds=self.bounds,options={'maxiter':1e7,'gtol': 1e-16,'ftol':1e-16, 'disp': True})
        self.GP.covarFunction.parameters.values =  optimizer.x
        self.GP.covarFunction.print_info()
        
    def nested_sampling(self,priors,resume=True,logdir=None):
        nested = ultranest.ReactiveNestedSampler(self.GP.covarFunction.parameters.names, self.GP.wrapperLogMarginalLikelihood,priors,resume=resume,log_dir=logdir)
        results = nested.run()
        nested.plot()
        self.GP.covarFunction.parameters.values =  results['posterior']['median']
        return results 
    