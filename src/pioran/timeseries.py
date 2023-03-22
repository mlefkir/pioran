"""Generic class and functions for fake time series.
"""
from numpy import savetxt
import jax.numpy as jnp
from jax import random
from jax.scipy.linalg import cholesky

from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import warnings

from .psd_base import PowerSpectralDensity
from .acvf_base import CovarianceFunction
from .utils import EuclideanDistance



class Simulations: 
    """Class to simulate time series from a given PSD or ACVF.
    
    
    Keys to generate the random numbers in the simulations:
        

    Parameters
    ----------
    T : :obj:`float`
        duration of the time series.
    dt : :obj:`float`
        sampling period of the time series.
    S_low : :obj:`float`
        Scale factor for the lower bound of the frequency grid.
    S_high : :obj:`float`
        Scale factor for the upper bound of the frequency grid.
    model : PowerSpectralDensity or CovarianceFunction
        The model for the PSD or ACVF.

    Attributes
    ----------
    duration : :obj:`float`
        duration of the time series.
    sampling_period : :obj:`float`
        sampling period of the time series.
    n_time : :obj:`int`
        number of time indexes.
    t : jnp.ndarray
        time :obj:`jnp.array` of the time series.
    f_max_obs : :obj:`float` 
        maximum frequency of the observed frequency grid.
    f_min_obs : :obj:`float`
        minimum frequency of the observed frequency grid.
    f0 : :obj:`float`
        minimum frequency of the total frequency grid.
    fN : :obj:`float`
        maximum frequency of the total frequency grid.
    n_freq_grid : :obj:`int`
        number of frequency indexes.
    frequencies : jnp.ndarray
        frequency :obj:`jnp.array` of the total frequency grid.
    tau_max : :obj:`float`
        maximum lag of the autocovariance function.
    dtau : :obj:`float`
        sampling period of the autocovariance function.
    tau : jnp.ndarray
        lag :obj:`jnp.array` of the autocovariance function.
    psd : jnp.ndarray
        power spectral density of the time series.
    acvf : jnp.ndarray
        autocovariance function of the time series.
    triang : jnp.ndarray
        triangular matrix used to generate the time series with the Cholesky decomposition.
    keys : dict
        dictionary of the keys used to generate the random numbers.
        'simu_TS' : key for drawing the values of the time series.
        'errors' : key for drawing the size of the errorbar of the time series from a given distribution.
        'fluxes' : key for drawing the fluxes of the time series from a given distribution.
        'subset' : key for randomising the choice of the subset of the time series.
        'sampling' : key for randomising the choice of the sampling of the time series.
        
    Methods
    -------
    generate_keys(seed)
        Generate the keys for the random numbers.
    plot_psd(figsize,filename)
        Plot the PSD of the time series.
    plot_acvf(figsize,filename)
        Plot the ACVF of the time series.
    autocovariance_method()
        Generate the time series with the autocovariance method.
    Timmer_Koenig_method()
        Generate the time series with the Timmer-Koenig method.
    sample_time_series(t,y,M,irregular_sampling=False)
        Sample the timeseries.
    extract_subset_timeseries(t,y,M)
        Extract a subset of the time series.
    simulate(mean=None,variance=None,method='ACV',irregular_sampling=False,randomise_fluxes=True,errors='gauss',seed=0,filename=None,**kwargs)  
        Simulate a time series.
    
    Raises
    ------
    ValueError
        If the model is not a PSD or ACVF.
    """
    
    def __init__(self, T, dt, S_low,S_high,model):

        # parameters of the time series
        self.duration = T
        self.sampling_period = dt
        self.n_time = jnp.rint(T/dt).astype(int)
        self.t = jnp.arange(0,self.duration,self.sampling_period)      
        # print(f"Number of time indexes desired : {self.n_time}")
         
        # parameters of the **observed** frequency grid 
        self.f_max_obs = 0.5/dt
        self.f_min_obs = 1/T
        
        # parameters of the **total** frequency grid
        self.f0 = self.f_min_obs/S_low
        self.fN = self.f_max_obs*S_high
        self.n_freq_grid = jnp.rint(jnp.ceil(self.fN/self.f0)).astype(int) + 1 
        # print(f"Number of frequency indexes desired : {self.n_freq_grid}")
        self.frequencies = jnp.arange(0,self.fN+self.f0,self.f0)
        self.tau_max = .5/self.f0#0.5/self.f0
        self.dtau = self.tau_max/(self.n_freq_grid-1) 
        self.tau = jnp.arange(0,self.tau_max+self.dtau,self.dtau)
        
        self.psd = None
        self.triang = None
        self.acvf = None
        self.keys = {}
            
        if isinstance(model,PowerSpectralDensity):
            self.psd = model.calculate(self.frequencies)
        elif isinstance(model,CovarianceFunction):
            self.acvf = model.calculate(self.tau)
        else:
            raise ValueError(f"You must provide a model which is either a PowerSpectralDensity or a CovarianceFunction, not {type(model)}")
                
        
    def generate_keys(self,seed=0):
        """Generate the keys to generate the random numbers.
        
        This function generates the keys to generate the random numbers for the simulations and store them in the dictionary self.keys.
        The keys and their meaning are:
        
        'simu_TS' : key for drawing the values of the time series.
        'errors' : key for drawing the size of the errorbar of the time series from a given distribution.
        'fluxes' : key for drawing the fluxes of the time series from a given distribution.
        'subset' : key for randomising the choice of the subset of the time series.
        'sampling' : key for randomising the choice of the sampling of the time series.
        
        Parameters
        ----------
        seed  : :obj:`int`, optional
            Seed for the random number generator, by default 0
        """
        
        key = random.PRNGKey(seed)
        self.keys['ts'],self.keys['errors'],self.keys['fluxes'],self.keys['subset'],self.keys['sampling'] = random.split(key,5)

        
    def plot_acvf(self,figsize=(15,3),filename=None):
        """Plot the autocovariance function.
        
        Plot the autocovariance function of the time series.
        
        Parameters
        ----------
        figsize  : :obj:`tuple`, optional
            Size of the figure, by default (15,3)
        filename  : :obj:`str`, optional
            Name of the file to save the figure, by default None
        
                
        Returns
        -------
        fig: :obj:`matplotlib.figure.Figure`
            Figure of the plot
        ax: :obj:`matplotlib.axes.Axes`
            Axes of the plot
        
        """
        
        if self.acvf is None:
            raise NotImplementedError("Plotting the PSD when the model is the ACV is not implemented yet")
            # acv = jnp.fft.irfft(self.psd)
            # self.acvf = acv[:len(acv)//2+1]/self.dtau
            
        fig,ax = plt.subplots(1,1,figsize=figsize)
        ax.plot(self.tau,self.acvf,'.-')
        ax.legend()
        ax.set_xlim(0,self.duration)
        ax.set_xlabel(r'Time lag $\tau (\mathrm{day})$')
        ax.set_ylabel('ACVF')
        ax.set_title("A model for the Autocovariance function")
        fig.tight_layout()
        
        if filename is not None:
            fig.savefig(f'{filename}',format='pdf')
        return fig,ax
            
    def plot_psd(self,figsize=(15,3),filename=None):
        """Plot the power spectral density model.
        
        A plot of the power spectral density model is generated.
        
        Parameters
        ----------
        figsize  : :obj:`tuple`, optional
            Size of the figure, by default (15,3)
        filename  : :obj:`str`, optional
            Name of the file to save the figure, by default None
        
        Returns
        -------
        fig: :obj:`matplotlib.figure.Figure`
            Figure of the plot
        ax: :obj:`matplotlib.axes.Axes`
            Axes of the plot
        
        """
        
        if self.psd is None:
            raise NotImplementedError("Plotting the PSD when the model is the ACV is not implemented yet")
        
        fig,ax = plt.subplots(1,1,figsize=figsize)
        ax.plot(self.frequencies,self.psd,'.-')
        ax.vlines(self.f_max_obs,ymin=jnp.min(self.psd),ymax=jnp.max(self.psd),label=r"$f_{\rm max}$",color='red')
        ax.vlines(self.f_min_obs,ymin=jnp.min(self.psd),ymax=jnp.max(self.psd),label=r"$f_{\rm min}$",color='g')
        ax.loglog()
        ax.legend()
        ax.set_xlabel(r'Frequency $(\mathrm{day}^{-1})$')
        ax.set_ylabel("PSD")
        ax.set_title("A model for the power spectral density")
        fig.tight_layout()
        
        if filename is not None:
            fig.savefig(f'{filename}',format='pdf')
        return fig,ax
        
        
        
    def autocovariance_method(self,interpolation='cubic'):
        """Generate a time series using the autocovariance method
        
        If the ACVF is not already calculated, it is calculated from the PSD 
        using the inverse Fourier transform.
        
        Returns
        -------
        :obj:`jnp.ndarray`
            Time array of the time series.
        :obj:`jnp.ndarray`
            Time series.
        
        """
        
        # if no acvf is given, we calculate it from the psd
        if self.acvf is None:
            acv = jnp.fft.irfft(self.psd)
            self.acvf = acv[:len(acv)//2+1]/self.dtau
        
        
        t_test = jnp.arange(0,self.duration,self.sampling_period)      

        # if the cholesky decomposition of the autocovariance function is not already calculated, we calculate it
        if self.triang is None:
            if interpolation == 'linear':
                interpo = interp1d(self.tau,self.acvf,'linear')
            elif interpolation == 'cubic':
                interpo = interp1d(self.tau,self.acvf,'cubic')
            dist = EuclideanDistance(t_test.reshape(-1,1),t_test.reshape(-1,1))
            K = interpo(dist)
            self.triang = cholesky(K)
        
        r = random.normal(key=self.keys['ts'],shape=(len(t_test),))
        ts = self.triang.T@r
        return t_test,ts
    
    
    def simulate(self, mean=None,variance=None,method='ACV',irregular_sampling=False,randomise_fluxes=True,errors='gauss',seed=0,filename=None,**kwargs):
        """Method to simulate time series using either the ACV method or the TK method.
        
        When using the ACV method, the time series is generated using an analytical autocovariance function or a power spectral density.
        If the autocovariance function is not provided, it is calculated from the power spectral density using the inverse Fourier transform
        and interpolated using a linear interpolation to map the autocovariance function on a grid of time lags.
        
        When using the TK method, the time series is generated using the Timmer and Koenig method for a larger duration and then the final time series
        is obtained by taking a subset of the generate time series.
        
        If the irregular_sampling flag is set to True, the time series will be sampled at irregular time intervals randomly.
        

        Parameters
        ----------
        mean : :obj:`float`, optional
            Mean of the time series, if None the mean will be set to -2 min(ts)
        method : :obj:`str`, optional
            method to simulate the time series, by default 'ACV' 
            can be 'TK' which uses Timmer and Koening method
        randomise_fluxes : bool, optional
            If True the fluxes will be randomised.
        errors : :obj:`str`, optional
            If 'gauss' the errors will be drawn from a gaussian distribution
        irregular_sampling : bool, optional
            If True the time series will be sampled at irregular time intervals
        seed : :obj:`int`, optional
            Set the seed for the RNG
        filename : :obj:`str`, optional
            Name of the file to save the time series, by default None
        **kwargs : :obj:`dict`
            Additional arguments to pass to the method
                interp_method : :obj:`str`, optional
                    Interpolation method to use when calculating the autocovariance function from the power spectral density, by default 'linear'
        

        Raises
        ------
        ValueError
            If the method is not 'ACV' or 'TK'
        ValueError
            If the errors are not 'gauss' or 'poisson'
        
        Returns
        -------
        t : jnp.ndarray
            Time :obj:`jnp.array`
        ts : jnp.ndarray
            Simulated time series
        ts_err : jnp.ndarray
            Errors on the simulated time series
        
        """
        self.generate_keys(seed=seed)
        
        if method == 'ACV':
            if self.n_time > 8000:
                warnings.warn("The desired number of point in the simulated time series is quite high")
            interp_method = kwargs.get('interp_method','cubic')
            t,true_timeseries = self.autocovariance_method(interpolation=interp_method)

        elif method == 'TK':
            if self.psd is None:
                raise ValueError("You must provide a PowerSpectralDensity model to use the TK method")
            # generate a long time series using the TK method
            long_t,long_true_timeseries = self.Timmer_Koenig_method()
            # minimal index of the start of the subset of the time series
            start_subset = jnp.rint(self.duration/(0.5/self.fN)).astype(int)
            subset_t, subset_true_timeseries = self.extract_subset_timeseries(long_t,long_true_timeseries,start_subset)
            t, true_timeseries = self.sample_timeseries(subset_t,subset_true_timeseries,self.n_time,irregular_sampling=irregular_sampling)  
            t -= t[0]
            
        else:
            raise ValueError(f"method {method} is not accepted, use either 'ACV' or 'TK'")
        #(rate-avg)/std * mean * rms + mean
        old_std = jnp.std(true_timeseries)
        old_mean = jnp.mean(true_timeseries)
        # rescale the time series to have the desired variance
        if variance is not None and method != 'ACV':
            true_timeseries = (true_timeseries - old_mean) / old_std * jnp.sqrt(variance)
        elif method != 'ACV':
            true_timeseries = (true_timeseries - old_mean) / old_std * jnp.sqrt(self.variance) 
        
            
        if randomise_fluxes:
            if errors == 'gauss':
                # generate the variance of the errors
                timeseries_error_size = jnp.abs(random.normal(key=self.keys['errors'],shape=(len(t),)))
                # generate the measured time series with the associated fluxes
                observed_timeseries = true_timeseries + timeseries_error_size*random.normal(key=self.keys['fluxes'],shape=(len(t),))
            elif errors == 'poisson':
                raise NotImplementedError("Poisson errors are not implemented yet")
                # observed_timeseries = random.poisson(key=self.keys['errors'],lam=true_timeseries,shape=(len(true_timeseries),))
                # timeseries_error_size = jnp.sqrt(observed_timeseries)
            else:
                raise ValueError(f"Error type {errors} is not accepted, use either 'gauss' or 'poisson'")
          
        else:
            timeseries_error_size = jnp.zeros_like(t)
            observed_timeseries = true_timeseries

        if mean is not None:
            observed_timeseries = observed_timeseries - jnp.mean(observed_timeseries) + mean
        else:
            observed_timeseries = observed_timeseries - 2*jnp.min(observed_timeseries)
         
        if filename is not None:
            savetxt(filename,jnp.vstack([t,observed_timeseries,timeseries_error_size]).T)
        return t,observed_timeseries,timeseries_error_size
        
    def extract_subset_timeseries(self,t,y,M):
        """Select a random subset of points from an input time series.
        
        The input time series is regularly sampled of size N. 
        The output time series is of size M with the same sampling rate as the input time series.
            
        Parameters
        ----------
        t : :obj:`jnp.array`
            Input time series of size N.
        y : :obj:`jnp.array`
            The fluxes of the simulated light curve.
        M : :obj:`int`
            The number of points in the desired time series.

        Returns
        -------
        :obj:`jnp.array`
            The time series of size M.
        :obj:`jnp.array`
            The values of the time series of size M.
        """
        N = len(t)
        start_index = random.randint(key=self.keys['subset'],shape=(1,),minval=M,maxval=N-M-1)[0]
        return t[start_index:start_index+M],y[start_index:start_index+M]


    def sample_timeseries(self,t,y,M,irregular_sampling=False):
        """Extract a random subset of points from the time series.
        
        Extract a random subset of M points from the time series. The input time series t is regularly sampled of size N with a sampling period dT.
        If irregular_sampling is False, the output time series has a sampling period dT/M.
        If irregular_sampling is True, the output time series is irregularly sampled.
        
        Parameters
        ----------
        t : :obj:`jnp.array`
            The time indexes of the time series.
        y : :obj:`jnp.array`
            The values of the time series.
        M : :obj:`int`
            The number of points in the desired time series.
        irregular_sampling : bool
            If True, the time series is irregularly sampled. If False, the time series is regularly sampled.
        
        Returns
        -------
        :obj:`jnp.array`
            The time indexes of the sampled time series.
        :obj:`jnp.array`
            The values of the sampled time series.
        """
        input_sampling_period = t[1]-t[0]
        output_sampling_period = (t[-1]-t[0])/M
    
        if not irregular_sampling:
            index_step_size = jnp.rint(output_sampling_period/input_sampling_period).astype(int)
            return t[::index_step_size],y[::index_step_size]
        else:
            t_sampled = random.choice(key=self.keys['sampling'],a=t, shape=(M,), replace=False)
            y_sampled = random.choice(key=self.keys['sampling'],a=y, shape=(M,), replace=False)
            sorted_indices = jnp.argsort(t_sampled)
            
            return t_sampled[sorted_indices],y_sampled[sorted_indices]

            
    def Timmer_Koenig_method(self):
        r"""Generate a time series using the Timmer-Konig method.
        
        Use the Timmer-Konig method to generate a time series with a given power spectral density
        stored in the attribute psd. 
        
        Assuming a power-law shaped PSD, the method is as follows:
        Draw two independent Gaussian random variables with zero mean and unit variance.
        The random variables are drawn using the key self.keys['ts'] split into two subkeys.
        
            1. Define A = sqrt(PSD/2) * (N1 + i*N2)
            2. Define A[0] = 0
            3. Define A[-1] = real(A[-1])
            4. ts = irfft(A)
            5. t is defined as the time indexes of the time series, with a sampling period of 0.5/fN.
        
        The duration of the time series is 2*(len(psd)-1).

        Returns
        -------
        :obj:`jnp.array`
            The time indexes of the time series.
        :obj:`jnp.array`
            The values of the time series.
        
        """
        # split the key into two subkeys
        key,subkey = random.split(self.keys['ts']) 
        
        N1, N2 = random.normal(key=key,shape=(len(self.psd),)), random.normal(key=subkey,shape=(len(self.psd),))
        randpsd = jnp.sqrt(0.5*self.psd) * ( N1 + 1j * N2 )
        randpsd.at[0].set(0)
        randpsd.at[-1].set(jnp.real(randpsd[-1]))
        
        # t = jnp.linspace(0,1/self.f0,2*(len(self.psd)-1))
        ts = jnp.fft.irfft(randpsd)
        dt = 0.5/self.fN
        t = jnp.arange(0,dt*len(ts),dt)
        self.variance = jnp.sum(self.psd*self.f0)
        return t,ts
