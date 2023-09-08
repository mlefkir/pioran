import os
import sys
from typing import Union

import jax.numpy as jnp
import numpy as np

from .carma_core import CARMAProcess
from .core import GaussianProcess
from .plots import (plot_posterior_predictive_ACF,
                    plot_posterior_predictive_PSD, plot_prediction,
                    plot_residuals)
from .psdtoacv import PSDToACV
from .utils.carma_utils import (CARMA_autocovariance, CARMA_powerspectrum,
                                MA_quad_to_coeff, quad_to_coeff)

from .utils.psd_utils import SHO_power_spectrum


class Visualisations:
    """Class for visualising the results after an inference run.
    
    Parameters
    ----------
    process : Union[GaussianProcess,CARMAProcess]
        The process to be visualised.
    filename : str
        The filename prefix for the output plots.
    **kwargs
        Additional keyword arguments.
        
    Attributes
    ----------
    process : Union[GaussianProcess,CARMAProcess]
        The process to be visualised.
    x : jnp.ndarray
        The observation times.
    y : jnp.ndarray
        The observation values.
    yerr : jnp.ndarray
        The observation errors.
    predictive_mean : jnp.ndarray
        The predictive mean.
    predictive_cov : jnp.ndarray
        The predictive covariance.
    x_pred : jnp.ndarray
        The prediction times.
    
    
    """
    def __init__(self,process: Union[GaussianProcess,CARMAProcess],filename,**kwargs) -> None:
        self.process = process
        
        self.x = process.observation_indexes.flatten()
        self.y = process.observation_values.flatten()
        self.yerr = process.observation_errors.flatten()

        self.f_min = 1/(self.x[-1]-self.x[0])
        self.f_max = .5/jnp.min(jnp.diff(self.x))
        n_frequencies = kwargs.get("n_frequencies",2500)
        self.frequencies = jnp.logspace(jnp.log10(self.f_min),jnp.log10(self.f_max),n_frequencies)
        self.tau = jnp.linspace(0,self.x[-1],1000)
        self.filename_prefix = filename
        
    def plot_timeseries_diagnostics(self,prediction_indexes=None,**kwargs) -> None:
        """Plot the timeseries diagnostics.

        This function will call the :func:`plot_prediction` and :func:`plot_residuals` functions to 
        plot the predicted timeseries and the residuals.

        """
        print("Plotting timeseries diagnostics...")
        log_transform = False if (not self.process.use_tinygp and self.process.log_transform) else None
        
        self.predictive_mean, self.predictive_cov = self.process.compute_predictive_distribution(log_transform=log_transform,prediction_indexes=prediction_indexes)
        self.x_pred = self.process.prediction_indexes.flatten() if prediction_indexes is None else prediction_indexes          
        fig,ax = plot_prediction(x = self.x.flatten(),
                                 y = self.y.flatten(),
                                 yerr = self.yerr.flatten(),
                                 x_pred = self.x_pred.flatten(),
                                 y_pred = self.predictive_mean.flatten(),
                                 cov_pred = self.predictive_cov,
                                 filename = self.filename_prefix,
                                 log_transform = self.process.log_transform,
                                 **kwargs)
        
        prediction_at_observation_times, _ = self.process.compute_predictive_distribution(log_transform=log_transform, prediction_indexes=self.x)
        
        fig2,ax2 = plot_residuals(x=self.x.flatten(),
                                  y=self.y.flatten(),
                                  yerr=self.yerr.flatten(),
                                  y_pred=prediction_at_observation_times.flatten(),
                                  filename=self.filename_prefix,
                                  log_transform=self.process.log_transform,
                                  **kwargs)
        
    def posterior_predictive_checks(self,samples,plot_PSD=True,plot_ACVF=True,**kwargs):
        """Plot the posterior predictive checks.

        Parameters
        ----------
        samples : jnp.ndarray
            The samples from the posterior distribution.
        plot_PSD : bool, optional
            Plot the posterior predictive PSDs, by default True
        plot_ACVF : bool, optional
            Plot the posterior predictive ACVFs, by default True
        **kwargs
            Additional keyword arguments.
            frequencies : jnp.ndarray, optional
                The frequencies at which to evaluate the PSDs of CARMA process, by default self.frequencies
            plot_lombscargle : bool, optional
                Plot the Lomb-Scargle periodogram, by default False
        """
        if isinstance(self.process,CARMAProcess):
            if self.process.p >1:
            
                print('Converting CARMA samples to coefficients...')
                alpha = [quad_to_coeff(samples[i,1:self.process.p+1]) for i in range(samples.shape[0])]
                sigma = samples[:,0]
                roots = [jnp.unique(jnp.roots(alpha[i]))[::-1] for i in range(samples.shape[0])]
                if self.process.q > 0:
                    if self.process.use_beta:
                        beta = samples[:,self.process.p+1:self.process.p+1+self.process.q]
                    else:
                        beta = [MA_quad_to_coeff(self.process.q,samples[i,self.process.p+1:self.process.p+1+self.process.q]) for i in range(samples.shape[0])]
            elif self.process.p == 1:
                alpha = samples[:,1]
                sigma = samples[:,0]
        else:
            if self.process.estimate_variance:
                index_variance = (self.process.model.parameters.free_names).index('var')                
                variance = samples[:,index_variance]
            if self.process.scale_errors:
                index_nu = (self.process.model.parameters.free_names).index('nu')                
                nu = samples[:,index_nu]
            if self.log_transform:
                index_const = (self.process.model.parameters.free_names).index('const')
                const = samples[:,index_const]
            params = samples
            
        
        posterior_ACVF = None
        # plot the posterior predictive PSDs
        if plot_PSD:
            print("Computing posterior predictive PSDs...")
            f = kwargs.get('frequencies',self.frequencies)
            if isinstance(self.process,CARMAProcess):
                if self.process.p == 1:
                    posterior_PSD = jnp.array([sigma[i]/(alpha[i]**2 + 4*jnp.pi**2*f) for i in range(samples.shape[0])])
                else:
                    if self.process.q > 0:
                        posterior_PSD = jnp.array([CARMA_powerspectrum(f,alpha[i],beta[i],sigma[i]) for i in range(samples.shape[0])])
                    else:
                        posterior_PSD = jnp.array([CARMA_powerspectrum(f,alpha[i],jnp.append(jnp.array([1]),jnp.zeros(self.process.p-1)),sigma[i]) for i in range(samples.shape[0])])
                print("Plotting posterior predictive PSDs...")
                plot_posterior_predictive_PSD(f=f,posterior_PSD=posterior_PSD,x=self.x,f_LS=self.frequencies,
                            y=self.y,yerr=self.yerr,filename=self.filename_prefix,save_data=True,**kwargs)
        
            else:
                f_min = self.process.model.f0 # 0 is the first frequency
                f_max = self.process.model.fN
                f = jnp.logspace(jnp.log10(f_min),jnp.log10(f_max),1000)
                                        
                posterior_PSD = []
                posterior_ACVF = []

                if isinstance(self.process.model,PSDToACV) and not self.process.use_tinygp:

                    if self.process.estimate_variance:
                        sumP = np.array([])
                        if not os.path.isfile(f'{self.filename_prefix}_normalisation_factor.txt'):
                            for it in range(samples.shape[0]):
                                
                                self.process.model.parameters.set_free_values(samples[it])
                                R, factor = self.process.model.calculate(self.tau,with_ACVF_factor=True)
                                sumP = np.append(sumP,factor)
                                # posterior_ACVF.append(R)
                                P = self.process.model.PSD.calculate(f)/factor*variance[it]

                                posterior_PSD.append(P) 
                                print(f'Sample: {it+1}/{samples.shape[0]}', end='\r')
                                sys.stdout.flush()
                            np.savetxt(f'{self.filename_prefix}_normalisation_factor.txt',sumP)

                        else:
                            print('Normalisation factor already computed, loading it...')
                            factors = np.loadtxt(f'{self.filename_prefix}_normalisation_factor.txt')
                            
                            for it in range(samples.shape[0]):
                                
                                self.process.model.parameters.set_free_values(samples[it])
                                P = self.process.model.PSD.calculate(f)/factors[it]*variance[it]#self.process.model.frequencies[1:])
                                posterior_PSD.append(P) 

                                print(f'Sample: {it+1}/{samples.shape[0]}', end='\r')
                                sys.stdout.flush()                    

                    else:
                        for it in range(samples.shape[0]):
                            self.process.model.parameters.set_free_values(samples[it])
                            posterior_PSD.append(self.process.model.PSD.calculate(f))  
                            print(f'Samples: {it+1}/{samples.shape[0]}', end='\r')
                            sys.stdout.flush()
                            
                    posterior_PSD = np.array(posterior_PSD)
                    f_LS = self.frequencies
                    
                    plot_posterior_predictive_PSD(f=f,posterior_PSD=posterior_PSD,x=self.x,
                                 y=np.log(self.y-np.mean(const)) if self.process.log_transform else self.y,
                                 yerr=self.yerr,filename=self.filename_prefix,save_data=True,
                                 f_LS=f_LS,f_min_obs=self.f_min,f_max_obs=self.f_max,**kwargs)
                
                # when the PSD model is approximated with tinygp, it is normalised to the variance
                elif isinstance(self.process.model,PSDToACV) and self.process.use_tinygp:
                    posterior_PSD_approx = []
                    if self.process.model.method != 'SHO':
                        raise NotImplementedError("Posterior predictive PSDs are not implemented for PSDToACV with tinygp.")
                    
                    print("Plotting posterior predictive PSDs with tinygp...")
                    fc = self.process.model.spectral_points
                    factors = []
                    for it in range(samples.shape[0]):
    
                        self.process.model.parameters.set_free_values(samples[it])
                        amps, _ = self.process.model.get_SHO_coefs()
                        factors.append(jnp.sum(amps*fc))
                        
                        psd = self.process.model.PSD.calculate(f) # calculate the PSD
                        psd_sho = SHO_power_spectrum(f,amps[...,None],fc[...,None]).sum(axis=0)
                        
                        psd /= psd[0] # normalise the PSD to the first value
                        psd_sho /= psd_sho[0] # normalise the PSD to the first value
                        
                        psd /= jnp.sum(amps*fc) # normalise the PSD to the sum of the amplitudes
                        psd_sho /= jnp.sum(amps*fc) # normalise the PSD to the sum of the amplitudes
                        
                        psd *= variance[it] # scale the PSD to the variance
                        psd_sho *= variance[it] # scale the PSD to the variance
                        
                        posterior_PSD.append(psd)
                        posterior_PSD_approx.append(psd_sho)
                        print(f'Samples: {it+1}/{samples.shape[0]}', end='\r')
                        sys.stdout.flush()
                        
                    factors = np.array(factors)
                    np.savetxt(f'{self.filename_prefix}_normalisation_factor.txt',factors)
                    posterior_PSD = np.array(posterior_PSD)
                    posterior_PSD_approx = np.array(posterior_PSD_approx)
                    f_LS = self.frequencies
                    
                    
                    plot_posterior_predictive_PSD(f = f,posterior_PSD=posterior_PSD,
                                 x = self.x,
                                 y = np.log(self.y) if self.process.log_transform else self.y,
                                 yerr = self.yerr,
                                 filename = self.filename_prefix,
                                 save_data = True,
                                 posterior_PSD_approx = posterior_PSD_approx,
                                 f_LS = f_LS,
                                 f_min_obs = self.f_min,
                                 f_max_obs = self.f_max,
                                 **kwargs)
                    
            # plot the posterior predictive PSDs

        # plot the posterior predictive ACFs
        if plot_ACVF :
            print("Computing posterior predictive ACFs...")
            if isinstance(self.process,CARMAProcess):
                if self.process.p == 1:
                    posterior_ACVF = jnp.array([.5*sigma[i]/alpha[i]*jnp.exp(-alpha[i]*self.tau) for i in range(samples.shape[0])])
                else:
                    if self.process.q > 0:
                        posterior_ACVF = jnp.array([CARMA_autocovariance(self.tau,roots[i],beta[i],sigma[i]) for i in range(samples.shape[0])])
                    else:
                        posterior_ACVF = jnp.array([CARMA_autocovariance(self.tau,roots[i],jnp.array([1]),sigma[i]) for i in range(samples.shape[0])])
            elif isinstance(self.process.model,PSDToACV):
                pass
                # if self.process.estimate_variance:
                #         posterior_ACVF = []
                #         for it in range(samples.shape[0]):
                #             self.process.model.parameters.set_free_values(samples[it])
                #             R = self.process.model.calculate(self.tau)
                #             posterior_ACVF.append(R/R[0]*variance[it])
                #             print(f'Samples: {it+1}/{samples.shape[0]}', end='\r')
                #             sys.stdout.flush()
                #         posterior_ACVF = np.array(posterior_ACVF)                
            else:
                raise NotImplementedError("Posterior predictive ACFs are not implemented for Gaussian processes.")
            
            posterior_ACVF /= posterior_ACVF[:,0,None]
            
            print("Plotting posterior predictive ACFs...")
            plot_posterior_predictive_ACF(tau=self.tau,acf=posterior_ACVF,
                                           x=self.x,y=self.y,filename=self.filename_prefix,save_data=True,**kwargs)