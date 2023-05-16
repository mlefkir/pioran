from typing import Union

import jax.numpy as jnp
import matplotlib.pyplot as plt

from .carma_core import CARMAProcess
from .core import GaussianProcess
from .plots import plot_prediction,plot_residuals,posterior_predictive_PSD
from .utils.carma_utils import quad_to_coeff,PowerSpectrum

class Visualisations:
    
    def __init__(self,process: Union[GaussianProcess,CARMAProcess],filename,**kwargs) -> None:
        self.process = process
        
        self.x = process.observation_indexes.flatten()
        self.y = process.observation_values.flatten()
        self.yerr = process.observation_errors.flatten()
        
        
        self.predictive_mean, self.predictive_cov = process.compute_predictive_distribution()
        self.x_pred = process.prediction_indexes.flatten()


        self.f_min = 1/(self.x[-1]-self.x[0])
        self.f_max = .5/jnp.min(jnp.diff(self.x))
        n_frequencies = kwargs.get("n_frequencies",2500)
        self.frequencies = jnp.logspace(jnp.log10(self.f_min),jnp.log10(self.f_max),n_frequencies)
           
        self.filename_prefix = filename
        
        
    def plot_timeseries_diagnostics(self,**kwargs) -> None:
   
        fig,ax = plot_prediction(x=self.x.flatten(),y=self.y.flatten(),yerr=self.yerr.flatten(),x_pred = self.x_pred.flatten(),
                        y_pred=self.predictive_mean.flatten(),cov_pred=self.predictive_cov,filename=self.filename_prefix,**kwargs)
        
        prediction_at_observation_times, _ = self.process.compute_predictive_distribution(prediction_indexes=self.x)
        
        fig2,ax2 = plot_residuals(x=self.x.flatten(),y=self.y.flatten(),yerr=self.yerr.flatten(),
                                  y_pred=prediction_at_observation_times.flatten(),
                                  filename=self.filename_prefix,**kwargs)
        
    def posterior_predictive_checks(self,samples,plot_PSD=True,plot_ACVF=True,**kwargs):
        
        if isinstance(self.process,CARMAProcess):
            alpha = [quad_to_coeff(samples[i,1:self.process.p+1]) for i in range(samples.shape[0])]
            beta = samples[:,self.process.p+1:self.process.p+1+self.process.q]
            sigma = samples[:,0]
            posterior_PSD = jnp.array([PowerSpectrum(self.frequencies,alpha[i],beta[i],sigma[i]) for i in range(samples.shape[0])])
            
        else:
            raise NotImplementedError("Posterior predictive checks are only implemented for CARMA processes.")
        
        if plot_PSD:
            posterior_predictive_PSD(f=self.frequencies,posterior_PSD=posterior_PSD,x=self.x,
                                 y=self.y,filename=self.filename_prefix,**kwargs)
        
        # if plot_ACVF: