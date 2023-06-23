"""Power spectral density to autocovariance function conversion.

Class to convert a power spectral density to an autocovariance function via the inverse Fourier transform.

"""
import equinox as eqx
import jax
import jax.numpy as jnp
from jax_finufft import nufft2

from .parameters import ParametersModel
from .psd_base import PowerSpectralDensity
from .utils.gp_utils import (EuclideanDistance, decompose_triangular_matrix,
                             reconstruct_triangular_matrix)


class PSDToACV(eqx.Module):
    """Class for the conversion of a power spectral density to an autocovariance function.
    
    Computes the autocovariance function from a power spectral density using the
    inverse Fourier transform and interpolates it on a grid of time lags.

    Parameters
    ----------
    PSD : :class:`~pioran.psd_base.PowerSpectralDensity`
        Power spectral density object.
    S_low : :obj:`float`
        Lower bound of the frequency grid.
    S_high : :obj:`float`
        Upper bound of the frequency grid.
    T : :obj:`float`
        Duration of the time series.
    dt : :obj:`float`
        Sampling period of the time series.

        
    Attributes
    ----------
    PSD : :class:`~pioran.psd_base.PowerSpectralDensity`
        Power spectral density object.
    parameters : :class:`~pioran.parameters.ParametersModel`
        Parameters of the power spectral density.
    frequencies : :obj:`jax.Array`
        Frequency grid.
    n_freq_grid : :obj:`int`
        Number of points in the frequency grid.
    f0 : :obj:`float`
        Lower bound of the frequency grid.
    fN : :obj:`float`
        Upper bound of the frequency grid.
    tau : :obj:`jax.Array`
        Time lag grid.
    dtau : :obj:`float`
        Time lag step.
    method : :obj:`str`
        Method used to compute the autocovariance function. Can be 'FFT' if the inverse Fourier transform is used or 'NuFFT' 
        if the non uniform Fourier transform is used.
    with_var : :obj:`bool`
        If True, the variance of the autocovariance function is estimated.
    
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
    method: str
    f0: float
    S_low: float
    S_high: float
    fN: float
    n_freq_grid: int    
    with_var: bool
    
    def __init__(self, PSD:PowerSpectralDensity, S_low:float, S_high:float,T, dt, estimate_variance=True,**kwargs):
        """Constructor of the PSDToACV class."""
        
        self.with_var = estimate_variance
        if not isinstance(PSD,PowerSpectralDensity):
            raise TypeError("PSD must be a PowerSpectralDensity object")
        if S_low<2:
            raise ValueError("S_low must be greater than 2")
        self.PSD = PSD
        self.parameters = PSD.parameters
        
        if self.with_var:
            self.parameters.append('var',1,True,hyperparameter=False)
        
        # parameters of the time series
        duration = T
        sampling_period = dt
        n_time = int(T/dt)
        
        # parameters of the **observed** frequency grid 
        f_max_obs = 0.5/dt
        f_min_obs = 1/T
        
        self.S_low = S_low
        self.S_high = S_high
        # parameters of the **total** frequency grid
        self.f0 = f_min_obs/self.S_low
        self.fN = f_max_obs*self.S_high
        self.n_freq_grid = int(jnp.ceil(self.fN/self.f0)) + 1 
        self.frequencies = jnp.arange(0,self.fN+self.f0,self.f0) #my biggest mistake...
        # self.frequencies = jnp.arange(self.f0,self.fN+self.f0,self.f0)
        tau_max = .5/self.f0#0.5/self.f0
        self.dtau = tau_max/(self.n_freq_grid-1) 
        self.tau = jnp.arange(0,tau_max+self.dtau,self.dtau)
        self.method = kwargs.get('method','FFT')
      
    def print_info(self):
        print("PSD to ACV conversion")
        print("Method: ",self.method)
        print('S_low: ',self.S_low)
        print('S_high: ',self.S_high)
        print('f0: ',self.f0)
        print('fN: ',self.fN)
        print('n_freq_grid: ',self.n_freq_grid)
        
    def calculate(self,t: jnp.array,**kwargs)-> jax.Array:
        """
        Calculate the autocovariance function from the power spectral density.
        
        The autocovariance function is computed by the inverse Fourier transform by 
        calling the method get_acvf_byFFT. The autocovariance function is then interpolated
        using the method interpolation.
        
        Parameters
        ----------
        t : :obj:`jax.Array`
            Time lags where the autocovariance function is computed.
        
        Returns
        -------
        :obj:`jax.Array`
            Autocovariance values at the time lags t.
        """
        if self.method == 'FFT':
            psd = self.PSD.calculate(self.frequencies[1:])
            if self.with_var: psd /= jnp.trapz(psd)*self.f0
            psd = jnp.insert(psd,0,0) # add a zero at the beginning to account for the zero frequency
            acvf = self.get_acvf_byFFT(psd)
            # normalize by the frequency step 
            # acvf = acvf[:len(acvf)//2+1]/self.dtau was not working properly for odd number of points
            if self.with_var:
                return  self.interpolation(t,acvf)*self.parameters['var'].value
            return  self.interpolation(t,acvf)
        
        elif self.method == 'NuFFT':
            N = 2*(self.n_freq_grid-1)
            k = jnp.arange(-N/2,N/2)*self.f0
            psd = self.PSD.calculate(k)+0j
            return  self.get_acvf_byNuFFT(psd,t*4*jnp.pi**2)
    
    def get_acvf_byNuFFT(self,psd: jnp.array,t: jnp.array)-> jax.Array:
        """Compute the autocovariance function from the power spectral density using the non uniform Fourier transform.

        Parameters
        ----------
        psd : :obj:`jax.Array`
            Power spectral density values.
        t : :obj:`jax.Array`
            Time lags where the autocovariance function is computed.

        Returns
        -------
        :obj:`jax.Array`
            Autocovariance values at the time lags t.
            
        """
        P = 2 * jnp.pi / self.f0
        return nufft2(psd, t/P).real * self.f0
    
    def get_acvf_byFFT(self, psd: jnp.array)-> jax.Array:
        """Compute the autocovariance function from the power spectral density using the inverse Fourier transform.

        Parameters
        ----------
        psd : :obj:`jax.Array`
            Power spectral density.

        Returns
        -------
        :obj:`jax.Array`
            Autocovariance function.
        """
        
        
        acvf = jnp.fft.irfft(psd) 
        acvf = acvf[:len(self.tau)]/self.dtau 
        return  acvf
    
    @eqx.filter_jit
    def interpolation(self, t: jnp.array, acvf: jnp.array)-> jax.Array:
        """Interpolate the autocovariance function at the points t.

        Parameters
        ----------
        t : :obj:`jax.Array`
            Points where the autocovariance function is computed.
        acvf : :obj:`jax.Array`
            Autocovariance values at the points tau.

        Returns
        -------
        :obj:`jax.Array`
            Autocovariance function at the points t.
        """
        I = jnp.interp(t,self.tau,acvf)
        return  I

    def get_cov_matrix(self, xq: jnp.array, xp: jnp.array)-> jax.Array:
        """Compute the covariance matrix between two arrays xq, xp.

        The term (xq-xp) is computed using the :func:`~pioran.utils.EuclideanDistance` function from the utils module.
        If the method used is 'NuFFT' and if the two arrays have the same shape, the covariance matrix is computed only on the unique values of the distance matrix
        using the :func:`~pioran.utils.decompose_triangular_matrix` and :func:`~pioran.utils.reconstruct_triangular_matrix` functions from the utils module.
        Otherwise, the covariance matrix is computed on the full distance matrix.

        Parameters
        ----------
        xq : (N,1) :obj:`jax.Array`
            First array.
        xp : (M,1) :obj:`jax.Array`
            Second array.

        Returns
        -------
        (N,M) :obj:`jax.Array`
            Covariance matrix.
        """
        # Compute the Euclidean distance between the query and the points
        dist = EuclideanDistance(xq, xp)

        if self.method == 'NuFFT': 
            if xq.shape == xp.shape:
                unique, reverse_indexes, triu_indexes, n = decompose_triangular_matrix(dist)
                avcf_unique = self.calculate(unique)    
                return reconstruct_triangular_matrix(avcf_unique, reverse_indexes, triu_indexes, n) 
            else:
                d = dist.flatten()
                return self.calculate(d).reshape(dist.shape)
        else:
            # Compute the covariance matrix
            return self.calculate(dist)

    def __str__(self) -> str:
        return f"PSDToACV\n{self.PSD.__str__()}"
    
    def __repr__(self) -> str:
        return self.__str__()
    