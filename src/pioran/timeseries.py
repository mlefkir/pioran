"""Generic class and functions for fake time series.
"""
import jax.numpy as jnp
from jax import random
from jax.scipy.linalg import cholesky
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

from .psd_base import PowerSpectralDensity
from .acvf_base import CovarianceFunction
from .utils import EuclideanDistance



class Simulations:
    
    """_summary_

    Parameters
    ----------
    T : _type_
        _description_
    dt : _type_
        _description_
    S_low : _type_
        _description_
    S_high : _type_
        _description_
    PowerSpectrum : PowerSpectralDensity, optional
        _description_, by default None
    Autocovariance : CovarianceFunction, optional
        _description_, by default None

    Attributes
    ----------
    t : _type_
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    
    def __init__(self, T, dt, S_low,S_high,PowerSpectrum:PowerSpectralDensity=None,Autocovariance:CovarianceFunction=None):

    
        # parameters of the time series
        self.duration = T
        self.sampling_period = dt
        self.n_time = int(T/dt)
        self.t = jnp.arange(0,self.duration,self.sampling_period)      

         
        # parameters of the **observed** frequency grid 
        self.f_max_obs = 0.5/dt
        self.f_min_obs = 1/T
        
        # parameters of the **total** frequency grid
        self.f0 = self.f_min_obs/S_low
        self.fN = self.f_max_obs*S_high
        self.n_freq_grid = int(jnp.ceil(self.fN/self.f0)) + 1 

        self.frequencies = jnp.arange(0,self.fN+self.f0,self.f0)
        self.tau_max = .5/self.f0#0.5/self.f0
        self.dtau = self.tau_max/(self.n_freq_grid-1) 
        self.tau = jnp.arange(0,self.tau_max+self.dtau,self.dtau)
            
        self.psd = None
        self.triang = None
        self.acvf = None

            
        if PowerSpectrum is not None:
            self.psd = PowerSpectrum.calculate(self.frequencies)
            # self.acvf = jnp.fft.irfft(self.psd)
        elif Autocovariance is not None:
            self.acvf = Autocovariance.calculate(self.tau)

            # self.psd = jnp.fft.rfft(self.acvf)
        else:
            raise ValueError("You must provide either a PowerSpectralDensity or a CovarianceFunction")
        
    def plot_acvf(self):
        """Plot the autocovariance function
        
        """
        
        if self.acvf is None:
            print("-------Calculating the ACVF from the PSD-------")
            acv = jnp.fft.irfft(self.psd)
            self.acvf = acv[:len(acv)//2+1]/self.dtau
            
        fig,ax = plt.subplots(1,1,figsize=(15,3))
        ax.plot(self.tau,self.acvf,'.-')
        ax.legend()
        ax.set_xlim(0,self.duration)
        ax.set_xlabel(r'Time lag $\tau (\mathrm{day})$')
        ax.set_ylabel('ACVF')
        ax.set_title("A model for the Autocovariance function")
        return fig,ax
    
    def plot_psd(self):
        """Plot the power spectral density
        
        """
        
        fig,ax = plt.subplots(1,1,figsize=(15,3))
        ax.plot(self.frequencies,self.psd,'.-')
        ax.vlines(self.f_max_obs,ymin=jnp.min(self.psd),ymax=jnp.max(self.psd),label=r"$f_{\rm max}$",color='red')
        ax.vlines(self.f_min_obs,ymin=jnp.min(self.psd),ymax=jnp.max(self.psd),label=r"$f_{\rm min}$",color='g')
        ax.loglog()
        ax.legend()
        ax.set_xlabel(r'Frequency $(\mathrm{day}^{-1})$')
        ax.set_ylabel("PSD")
        ax.set_title("A model for the power spectral density")
        return fig,ax
        
        
        
    def ACV_method(self,seed=0):
        """Generate a time series using the ACV method
        
        If the ACVF is not already calculated, it is calculated from the PSD 
        using the inverse Fourier transform.
        
        
        """
        
        if self.acvf is None:
            acv = jnp.fft.irfft(self.psd)
            self.acvf = acv[:len(acv)//2+1]
        
        
        t_test = jnp.arange(0,self.duration,self.sampling_period)      

        if self.triang is None:
            interpo = interp1d(self.tau,self.acvf,'cubic')
            dist = EuclideanDistance(t_test.reshape(-1,1),t_test.reshape(-1,1))
            K = interpo(dist)
            self.triang = cholesky(K)
        
        key = random.PRNGKey(seed)
        r = random.normal(key,shape=(len(t_test),1)).flatten()
        ts = self.triang.T@r
        return t_test,ts
        
    def save_time_series(self,name,mean=None,seed=0):
        """Generate a time series and save it to a file.
        
        if mean is not None, the time series is shifted by twice the minimum of the time to have a positive valued time series.
        
        

        Parameters
        ----------
        name : _type_
            _description_
        mean : _type_, optional
            _description_, by default None
        seed : int, optional
            _description_, by default 0

        Returns
        -------
        _type_
            _description_
        """
        
        t,true_timeseries = self.ACV_method(seed)
        
        if mean is not None:
            true_timeseries += mean
        else:
            true_timeseries -= 2*jnp.min(true_timeseries)
            
            
        key = random.PRNGKey(seed+1)
        key,subkey = random.split(key) 

        # generate the variance of the errors
        timeseries_error_size = jnp.abs(random.normal(key,shape=(len(t),1)).flatten())
        # generate the measured time series with the associated fluxes
        observed_timeseries = true_timeseries + timeseries_error_size*random.normal(subkey,shape=(len(t),1)).flatten()
            
        np.savetxt(f"{name}_seed{seed}.txt",np.array([t,observed_timeseries,timeseries_error_size]).T)
        return t,observed_timeseries,timeseries_error_size
        
    def TimmerKonig(self,seed=0):
        
        
        key = random.PRNGKey(seed)
        key,subkey = random.split(key) 
               
        randpsd = jnp.sqrt(0.5*self.psd)*(random.normal(key,shape=(1,len(self.psd))) + 1j*random.normal(subkey,shape=(1,len(self.psd))))
        fullpsd = randpsd.flatten()
        fullpsd.at[0].set(0)
        fullpsd.at[-1].set(jnp.real(fullpsd[-1]))
        
        t = jnp.linspace(0,1/self.f0,2*(len(self.psd)-1))
        ts = jnp.fft.irfft(fullpsd)
        return t,ts.flatten()
