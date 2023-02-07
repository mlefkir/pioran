"""Power spectral density to autocovariance function conversion.

Class to convert a power spectral density to an autocovariance function via the inverse Fourier transform.

"""


import jax.numpy as jnp

from .psd_base import PowerSpectralDensityComponent, PowerSpectralDensity
from scipy.interpolate import interp1d
from .utils import EuclideanDistance
from typing import Union



class PSDToACV:
    
    def __init__(self, PSD:PowerSpectralDensity, S_low:float, S_high:float,T, dt):
        """
        Parameters
        ----------
        psd : PowerSpectralDensity
            Power spectral density object.
        S_low : float
            Lower bound of the frequency grid.
        S_high : float
            Upper bound of the frequency grid.
        T : float
            Duration of the time series.
        dt : float
            Sampling period of the time series.
        n_freq_grid : int, optional
            Number of points in the frequency grid, by default None
        """
        if not (isinstance(PSD,PowerSpectralDensity) or isinstance(PSD,PowerSpectralDensityComponent)):
            raise TypeError("PSD must be a PowerSpectralDensity object")
        if S_low<2:
            raise ValueError("S_low must be greater than 2")
        self.PSD = PSD
        self.parameters = PSD.parameters
        
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
        
    def calculate(self,x):
        """
        Calculate the autocovariance function from the power spectral density.
        """
        psd = self.PSD.calculate(self.frequencies)
        acvf = jnp.fft.irfft(psd) 
        acvf = acvf[:len(self.tau)]/self.dtau # normalize by the frequency step 
        # acvf = acvf[:len(acvf)//2+1]/self.dtau was not working properly for odd number of points

        interpo = interp1d(self.tau,acvf,'linear')
        CovMat = interpo(x)

        return CovMat
    
    def __str__(self) -> str:
        return f"PSDToACV({self.PSD.__str__()})"
    
    def get_cov_matrix(self, xq, xp):
        """Compute the covariance matrix between two arrays for the exponential covariance function.

        K(xq,xp) = variance * exp( - (xq-xp) / lengthscale )

        The term (xq-xp) is computed using the Euclidean distance from the module covarfun.distance

        Parameters
        ----------
        xq: array of shape (n,1)
            First array.
        xp: array  of shape (m,1)
            Second array.

        Returns
        -------
        K: array of shape (n,m)
            Covariance matrix.
        """
        # Compute the Euclidean distance between the query and the points
        dist = EuclideanDistance(xq, xp)
        # Compute the covariance matrix
        covMat = self.calculate(dist)

        return covMat