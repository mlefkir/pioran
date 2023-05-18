from typing import Union

import jax.numpy as jnp
import matplotlib.pyplot as plt

from .acvf import CARMA_covariance
from .acvf_base import CovarianceFunction
from .carma_core import CARMAProcess
from .core import GaussianProcess
from .plots import (plot_posterior_predictive_ACF,
                    plot_posterior_predictive_PSD, plot_prediction,
                    plot_residuals)
from .psd_base import PowerSpectralDensity
from .psdtoacv import PSDToACV
from .utils.carma_utils import Autocovariance, PowerSpectrum, quad_to_coeff


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
        self.tau = jnp.linspace(0,self.x[-1],1000)
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
            if self.process.p >1:
            
                print('Converting CARMA samples to coefficients...')
                alpha = [quad_to_coeff(samples[i,1:self.process.p+1]) for i in range(samples.shape[0])]
                sigma = samples[:,0]
                roots = roots = [jnp.roots(alpha[i])[::-1] for i in range(samples.shape[0])]
                if self.process.q > 0:
                    beta = samples[:,self.process.p+1:self.process.p+1+self.process.q]                
            elif self.process.p == 1:
                alpha = samples[:,1]
                sigma = samples[:,0]
        else:
            
            if self.process.scale_errors and self.process.estimate_mean:
                params = samples[:,:-2]
        
            elif self.process.scale_errors or self.process.estimate_mean:
                params = samples[:,:-1]
            else:
                params = samples
        
        # plot the posterior predictive PSDs
        if plot_PSD:
            print("Computing posterior predictive PSDs...")
            
            if isinstance(self.process,CARMAProcess):
                if self.process.p == 1:
                    posterior_PSD = jnp.array([sigma[i]/(alpha[i]**2 + 4*jnp.pi**2*self.frequencies) for i in range(samples.shape[0])])
                else:
                    if self.process.q > 0:
                        posterior_PSD = jnp.array([PowerSpectrum(self.frequencies,alpha[i],beta[i],sigma[i]) for i in range(samples.shape[0])])
                    else:
                        posterior_PSD = jnp.array([PowerSpectrum(self.frequencies,roots[i],jnp.append(jnp.array([1]),jnp.zeros(self.process.p-1)),sigma[i]) for i in range(samples.shape[0])])
                print("Plotting posterior predictive PSDs...")

            else:
                if isinstance(self.process.model,PSDToACV):
                    # posterior_PSD = jnp.array([self.process.model.PSD(self.frequencies,params[i]) for i in range(samples.shape[0])])
                    
                    raise NotImplementedError("Posterior predictive PSDs are not implemented for Gaussian processes.")
                
            plot_posterior_predictive_PSD(f=self.frequencies,posterior_PSD=posterior_PSD,x=self.x,
                                 y=self.y,filename=self.filename_prefix,**kwargs)
        
        # plot the posterior predictive ACFs
        if plot_ACVF:
            print("Computing posterior predictive ACFs...")
            if isinstance(self.process,CARMAProcess):
                if self.process.p == 1:
                    posterior_ACVF = jnp.array([.5*sigma[i]/alpha[i]*jnp.exp(-alpha[i]*self.tau) for i in range(samples.shape[0])])
                else:
                    if self.process.q > 0:
                        posterior_ACVF = jnp.array([Autocovariance(self.tau,roots[i],beta[i],sigma[i]) for i in range(samples.shape[0])])
                    else:
                        posterior_ACVF = jnp.array([Autocovariance(self.tau,roots[i],jnp.array([1]),sigma[i]) for i in range(samples.shape[0])])
            else:
                raise NotImplementedError("Posterior predictive ACFs are not implemented for Gaussian processes.")
            posterior_ACVF /= posterior_ACVF[:,0,None]
            
            print("Plotting posterior predictive ACFs...")
            plot_posterior_predictive_ACF(tau=self.tau,acf=posterior_ACVF,
                                           x=self.x,y=self.y,filename=self.filename_prefix,**kwargs)