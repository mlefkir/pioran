"""Power spectral density to autocovariance function conversion.

Class to convert a power spectral density to an autocovariance function via the inverse Fourier transform.

"""


import jax.numpy as jnp
import equinox as eqx

from .psd_base import PowerSpectralDensity
from .parameters import ParametersModel
from .utils import EuclideanDistance


class PSDToACV(eqx.Module):
    """Class for the conversion of a power spectral density to an autocovariance function.
    
    Computes the autocovariance function from a power spectral density using the
    inverse Fourier transform and interpolates it on a grid of time lags.

    Parameters
    ----------
    PSD : :obj:`PowerSpectralDensity`
        Power spectral density object.
    S_low : float
        Lower bound of the frequency grid.
    S_high : float
        Upper bound of the frequency grid.
    T : float
        Duration of the time series.
    dt : float
        Sampling period of the time series.

        
    Attributes
    ----------
    PSD : PowerSpectralDensity
        Power spectral density object.
    parameters : ParametersModel
        Parameters of the power spectral density.
    frequencies : jnp.ndarray
        Frequency grid.
    n_freq_grid : int
        Number of points in the frequency grid.
    f0 : float
        Lower bound of the frequency grid.
    fN : float
        Upper bound of the frequency grid.
    tau : jnp.ndarray
        Time lag grid.
    dtau : float
        Time lag step.
    
    Methods
    -------
    calculate(t)
        Calculate the autocovariance function from the power spectral density.
    get_acvf_byFFT(psd)
        Calculate the autocovariance function from the power spectral density using the inverse Fourier transform.
    interpolation(t)
        Interpolate the autocovariance function on a grid of time lags.
    
    """
    
    PSD: PowerSpectralDensity
    parameters: ParametersModel
    frequencies: jnp.ndarray
    tau: jnp.ndarray
    dtau: float
    
    def __init__(self, PSD:PowerSpectralDensity, S_low:float, S_high:float,T, dt):
        """Constructor of the PSDToACV class."""
        
        if not isinstance(PSD,PowerSpectralDensity):
            raise TypeError("PSD must be a PowerSpectralDensity object")
        if S_low<2:
            raise ValueError("S_low must be greater than 2")
        self.PSD = PSD
        self.parameters = PSD.parameters
        
        # parameters of the time series
        duration = T
        sampling_period = dt
        n_time = int(T/dt)
        
        # parameters of the **observed** frequency grid 
        f_max_obs = 0.5/dt
        f_min_obs = 1/T
        
        # parameters of the **total** frequency grid
        f0 = f_min_obs/S_low
        fN = f_max_obs*S_high
        n_freq_grid = int(jnp.ceil(fN/f0)) + 1 
        self.frequencies = jnp.arange(0,fN+f0,f0)
        tau_max = .5/f0#0.5/self.f0
        self.dtau = tau_max/(n_freq_grid-1) 
        self.tau = jnp.arange(0,tau_max+self.dtau,self.dtau)
      
    def calculate(self,t):
        """
        Calculate the autocovariance function from the power spectral density.
        
        The autocovariance function is computed by the inverse Fourier transform by 
        calling the method get_acvf_byFFT. The autocovariance function is then interpolated
        using the method interpolation.
        
        Parameters
        ----------
        t : array
            Time lags where the autocovariance function is computed.
        
        """
        psd = self.PSD.calculate(self.frequencies)
        acvf = self.get_acvf_byFFT(psd)
# normalize by the frequency step 
        # acvf = acvf[:len(acvf)//2+1]/self.dtau was not working properly for odd number of points

        return  self.interpolation(t,acvf)
    
    def get_acvf_byFFT(self,psd):
        """Compute the autocovariance function from the power spectral density using the inverse Fourier transform.

        Parameters
        ----------
        psd : array
            Power spectral density.

        Returns
        -------
        array
            Autocovariance function.
        """
        
        
        acvf = jnp.fft.irfft(psd) 
        acvf = acvf[:len(self.tau)]/self.dtau 
        return  acvf
    
    @eqx.filter_jit
    def interpolation(self, t, acvf):
        """Interpolate the autocovariance function at the points x.

        Parameters
        ----------
        t : array
            Points where the autocovariance function is computed.

        Returns
        -------
        array
            Autocovariance function at the points x.
        """
        # if kind=='linear':
        I = jnp.interp(t,self.tau,acvf)
        # else:
        #     interpo = interp1d(self.tau,acvf,'linear')
        #     I = interpo(x)
        return  I

    # @eqx.filter_jit
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
        return self.calculate(dist)

    def __str__(self) -> str:
        return f"PSDToACV\n{self.PSD.__str__()}"
    
    def __repr__(self) -> str:
        return self.__str__()
    