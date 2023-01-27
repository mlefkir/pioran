"""Generic class and functions for fake time series.
"""
from dataclasses import dataclass
import jax.numpy as jnp
from .parameters import ParametersModel
from .psd_base import PowerSpectralDensity
from .acvf_base import CovarianceFunction
from jax import random
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from .utils import EuclideanDistance
from jax.scipy.linalg import cholesky
import numpy as np
from jax.numpy.fft import irfft, rfft
import jax.scipy as jsp

class Simulations:
    
    def __init__(self, T, dt, S_low,S_high,PowerSpectrum:PowerSpectralDensity=None,Autocovariance:CovarianceFunction=None):

        # parameters of the time series
        self.duration = T
        self.sampling_period = dt
        self.n_time = int(T/dt)
         
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
        
    def save_time_series(self,name,errors,seed=0):
        t,ts = self.ACV_method(seed=0)
        if errors is not None:
            ts_err = np.sqrt(np.abs(ts)*self.duration)/self.duration
            np.savetxt(f"{name}_seed{seed}.txt",np.array([t,ts,ts_err]).T)
        else:
            np.savetxt(f"{name}_seed{seed}.txt",np.array([t,ts]).T)
        
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
