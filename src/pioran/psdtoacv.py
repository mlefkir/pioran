"""Power spectral density to autocovariance function conversion.

Class to convert a power spectral density to an autocovariance function via the inverse Fourier transform.

"""
import equinox as eqx
import jax
import jax.numpy as jnp
import tinygp
from jax_finufft import nufft2
from tinygp.kernels.quasisep import SHO as SHO_term

from .parameters import ParametersModel
from .psd_base import PowerSpectralDensity
from .utils.gp_utils import (EuclideanDistance, valid_methods,
                             decompose_triangular_matrix,
                             reconstruct_triangular_matrix)


class PSDToACV(eqx.Module):
    """Class for the conversion of a power spectral density to an autocovariance function.

    Computes the autocovariance function from a power spectral density using the several methods.

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
        Minimum sampling duration of the time series.
    method : :obj:`str`
        Method used to compute the autocovariance function. Can be 'FFT' if the inverse Fourier transform is used or 'NuFFT'
        for the non uniform Fourier transform. The 'SHO' method will approximate the power spectral density into a sum of SHO functions.
    n_components : :obj:`int`
        Number of components used to approximate the power spectral density using the 'SHO' method.
    estimate_variance : :obj:`bool`, optional
        If True, the amplitude of the autocovariance function is estimated. Default is True.
            
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
    f_max_obs : :obj:`float`
        Maximum observed frequency, i.e. the Nyquist frequency.
    f_min_obs : :obj:`float`
        Minimum observed frequency.
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
            for the non uniform Fourier transform. The 'SHO' method will approximate the power spectral density into a sum of SHO functions.
    n_components : :obj:`int`
            Number of components used to approximate the power spectral density using the 'SHO' method.
    estimate_variance : :obj:`bool`
        If True, the amplitude of the autocovariance function is estimated.
    spectral_points : :obj:`jax.Array`
        Frequencies of the SHO kernels.
    spectral_matrix : :obj:`jax.Array`
        Matrix of the SHO kernels.

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
    frequencies: jax.Array = None
    tau: jax.Array = None
    dtau: float = None
    method: str
    f_max_obs: float
    f_min_obs: float
    f0: float 
    S_low: float
    S_high: float
    fN: float
    n_freq_grid: int = None
    estimate_variance: bool
    n_components: int = None
    spectral_points: jax.Array = None
    spectral_matrix: jax.Array = None
    ACVF: tinygp.kernels.quasisep.SHO

    def __init__(self, PSD: PowerSpectralDensity, 
                 S_low: float, 
                 S_high: float, 
                 T: float, 
                 dt: float, 
                 method: str, 
                 n_components: int = None, 
                 estimate_variance = True):
        """Constructor of the PSDToACV class.
    
        Initialize the PSDToACV class with the power spectral density, the frequency grid and the method used to compute the autocovariance function.
    
    
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
            Minimum sampling duration of the time series.
        method : :obj:`str`
            Method used to compute the autocovariance function. Can be 'FFT' if the inverse Fourier transform is used or 'NuFFT'
            for the non uniform Fourier transform. The 'SHO' method will approximate the power spectral density into a sum of SHO functions.
        n_components : :obj:`int`
            Number of components used to approximate the power spectral density using the 'SHO' method.
        estimate_variance : :obj:`bool`, optional
            If True, the variance of the autocovariance function is estimated. Default is True.
        
        Raises
        ------
        TypeError
            If PSD is not a :class:`~pioran.psd_base.PowerSpectralDensity` object.
        ValueError
            If S_low is smaller than 2., if method is not in the allowed methods or if n_components is smaller than 1.
        """

        self.estimate_variance = estimate_variance
             
        # sanity checks:   
        if not isinstance(PSD, PowerSpectralDensity):
            raise TypeError(f"PSD must be a PowerSpectralDensity object, not a {type(PSD)}")
        
        if dt > T:
            raise ValueError(f"dt ({dt}) must be smaller than T ({T})")
        
        if S_low < 2:
            raise ValueError(f"S_low must be greater than 2, {S_low} was given")
        
        if method not in valid_methods:
            raise ValueError(f"Method {method} not allowed. Choose between {valid_methods}")
        
        if ('FFT' not in method ) and n_components < 1:
            raise ValueError("n_components must be greater than 1")
        
        
        # define the attributes
        self.PSD = PSD
        self.parameters = PSD.parameters
        self.method = method
        self.S_low = S_low  
        self.S_high = S_high

        if self.estimate_variance:
            self.parameters.append('var', 1, True, hyperparameter=False)

        # parameters of the **observed** frequency grid
        self.f_max_obs = 0.5/dt # Nyquist frequency
        self.f_min_obs = 1/T # minimum observed frequency

        # parameters of the **total** frequency grid
        self.f0 = self.f_min_obs/self.S_low # minimum frequency
        self.fN = self.f_max_obs*self.S_high # maximum frequency

        self.n_freq_grid = jnp.rint(jnp.ceil(self.fN/self.f0)) + 1
        self.frequencies = jnp.arange(0, self.fN+self.f0, self.f0)
            
        tau_max = 0.5/self.f0 
        self.dtau = tau_max/(self.n_freq_grid-1)
        self.tau = jnp.arange(0, tau_max+self.dtau, self.dtau)
        
        if self.method == 'FFT' or self.method == 'NuFFT':
            pass
        
        elif self.method == 'SHO':
            self.n_components = n_components
            self.spectral_points = jnp.geomspace(
                self.f0, self.fN, self.n_components)
            self.spectral_matrix = 1 / \
                (1 + jnp.power(jnp.atleast_2d(self.spectral_points).T /
                 self.spectral_points, 4))
        else:
            raise NotImplementedError(f"Method {self.method} not implemented")
        
    def decompose_model(self, psd_normalised: jax.Array):
        r"""Decompose the model into a sum of SHO kernels.

        Assuming that the model can be written as a sum of :math:`J` SHO kernels, this method
        solve the linear system to find the amplitude :math:`a_j` of each kernel.

        .. math:: :label: sho_power_spectrum 
    
        \boldsymbol{y} = B \boldsymbol{a}

        with :math:`\boldsymbol{y}=\begin{bmatrix}1 & \mathcal{P}(f_1)/\mathcal{P}(f_0) & \cdots & \mathcal{P}(f_J)/\mathcal{P}(f_0) \end{bmatrix}^\mathrm{T}` 
        the normalised power spectral density vector, :math:`B` the spectral matrix associated to the linear system and :math:`\boldsymbol{a}` the amplitudes of the SHO kernels.

        
        .. math:: :label: sho_spectral_matrix
        
        B_{ij} = \dfrac{1}{1 + \left(\dfrac{f_i}{f_j}\right)^4}
        

        Parameters
        ----------
        psd_normalised : :obj:`jax.Array`
            Normalised power spectral density by the first value.

        Returns
        -------
        amplitudes : :obj:`jax.Array`
            Amplitudes of the SHO kernels.
        frequencies : :obj:`jax.Array`
            Frequencies of the SHO kernels.        
        """

        a = jnp.linalg.solve(self.spectral_matrix, psd_normalised)
        return a, self.spectral_points

    def get_SHO_coefs(self):
        """Get the amplitudes and frequencies of the SHO kernels.
        
        Estimate the amplitudes and frequencies of the SHO kernels by solving the linear system.
        
        Returns
        -------
        amplitudes : :obj:`jax.Array`
            Amplitudes of the SHO kernels.
        frequencies : :obj:`jax.Array`
            Frequencies of the SHO kernels.        
        """
        psd = self.PSD.calculate(self.spectral_points)
        psd /= psd[0]

        a, f = self.decompose_model(psd)
        return a, f

    def build_SHO_model(self, amplitudes: jax.Array, frequencies: jax.Array)->tinygp.kernels.quasisep.SHO:
        """Build the semi-separable SHO model in tinygp from the amplitudes and frequencies.

        Parameters
        ----------
        amplitudes : :obj:`jax.Array`
            Amplitudes of the SHO kernels.
        frequencies : :obj:`jax.Array`
            Frequencies of the SHO kernels.

        Returns
        -------
        kernel : :obj:`tinygp.kernels.quasisep.SHO`        
            Constructed SHO kernel.
        """

        kernel = 0
        for j in range(self.n_components):
            kernel += amplitudes[j]*SHO_term(
                quality = 1/jnp.sqrt(2), 
                omega = 2*jnp.pi*frequencies[j])
        return kernel

    @property
    def ACVF(self)->tinygp.kernels.quasisep.SHO:
        """Get the autocovariance function from the SHO model.
        
        Define the autocovariance function from the semi-separable SHO model.
        This property is used to define the autocovariance function in the GP model.
        
        Returns
        -------
        :obj:`tinygp.kernels.quasisep.SHO`
            Autocovariance function as sum of SHO kernels.
        """
        psd = self.PSD.calculate(self.spectral_points)
        psd /= psd[0]

        a, f = self.decompose_model(psd)
        if self.method == 'SHO':
            kernel = self.build_SHO_model(a*f, f)
        else:
            raise NotImplementedError('Only SHO is implemented for now')
        if self.estimate_variance:
            return kernel*(self.parameters['var'].value/jnp.sum(a*f))
        return kernel

    def print_info(self):
        print("PSD to ACV conversion")
        print("Method: ", self.method)
        print('S_low: ', self.S_low)
        print('S_high: ', self.S_high)
        print('f0: ', self.f0)
        print('fN: ', self.fN)
        print('n_freq_grid: ', self.n_freq_grid)

    def calculate_rescale(self, t: jax.Array) -> jax.Array:

        if self.method == 'FFT':
            psd = self.PSD.calculate(self.frequencies[1:])
            # add a zero at the beginning to account for the zero frequency
            psd = jnp.insert(psd, 0, 0)
            acvf = self.get_acvf_byFFT(psd)
            if self.estimate_variance:
                # normalize by the variance instead of integrating the PSD with the trapezium rule
                return acvf[0]

    def calculate(self, t: jax.Array, with_ACVF_factor=False) -> jax.Array:
        """
        Calculate the autocovariance function from the power spectral density.

        The autocovariance function is computed by the inverse Fourier transform by 
        calling the method get_acvf_byFFT. The autocovariance function is then interpolated
        using the method interpolation.

        Parameters
        ----------
        t : :obj:`jax.Array`
            Time lags where the autocovariance function is computed.
        with_ACVF_factor : :obj:`bool`, optional
            If True, the autocovariance function is multiplied by the factor :math:`\mathcal{R}(0)`. Default is False.
            
        Raises
        ------
        NotImplementedError
            If the method is not implemented.

        Returns
        -------
        :obj:`jax.Array`
            Autocovariance values at the time lags t.
        """
        if self.method == 'FFT':
            psd = self.PSD.calculate(self.frequencies[1:])
            # add a zero at the beginning to account for the zero frequency
            psd = jnp.insert(psd, 0, 0)
            acvf = self.get_acvf_byFFT(psd)
            
            if self.estimate_variance:
                # normalize by the variance instead of integrating the PSD with the trapezium rule
                R = acvf / acvf[0]
                if not with_ACVF_factor:
                    return self.interpolation(t, R)*self.parameters['var'].value
                else:
                    return self.interpolation(t, R)*self.parameters['var'].value, acvf[0]
            return self.interpolation(t, acvf)

        elif self.method == 'NuFFT':
            
            if self.estimate_variance:
                raise NotImplementedError('estimate_variance not implemented for NuFFT')

            N = 2*(self.n_freq_grid-1)
            k = jnp.arange(-N/2, N/2)*self.f0
            psd = self.PSD.calculate(k)+0j
            return self.get_acvf_byNuFFT(psd, t*4*jnp.pi**2)

        else:
            raise NotImplementedError(f'Method {self.method} not implemented')

    def get_acvf_byNuFFT(self, psd: jax.Array, t: jax.Array) -> jax.Array:
        """Compute the autocovariance function from the power spectral density using the non uniform Fourier transform.

        This function uses the jax_finufft package to compute the non uniform Fourier transform with the nufft2 function.

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

    def get_acvf_byFFT(self, psd: jax.Array) -> jax.Array:
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
        return acvf

    @eqx.filter_jit
    def interpolation(self, t: jax.Array, acvf: jax.Array) -> jax.Array:
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
        I = jnp.interp(t, self.tau, acvf)
        return I

    def get_cov_matrix(self, xq: jax.Array, xp: jax.Array) -> jax.Array:
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
            
        Raises
        ------
        NotImplementedError
            If the method is not implemented.

        Returns
        -------
        (N,M) :obj:`jax.Array`
            Covariance matrix.
        """
        # Compute the Euclidean distance between the query and the points
        dist = EuclideanDistance(xq, xp)

        if self.method == 'NuFFT':
            if xq.shape == xp.shape:
                unique, reverse_indexes, triu_indexes, n = decompose_triangular_matrix(
                    dist)
                avcf_unique = self.calculate(unique)
                return reconstruct_triangular_matrix(avcf_unique, reverse_indexes, triu_indexes, n)
            else:
                d = dist.flatten()
                return self.calculate(d).reshape(dist.shape)
        elif self.method == 'FFT':
            # Compute the covariance matrix
            return self.calculate(dist)
        else:
            raise NotImplementedError(f"Calculating the covariance matrix for method '{self.method}' not implemented")

    def __str__(self) -> str:
        """String representation of the PSDToACV object.
        
        Returns
        -------
        :obj:`str`
            String representation of the PSDToACV object.
        """
        s = f"PSDToACV\n"
        if self.method != 'NuFFT' and self.method != 'FFT':
            s += f"method: {self.method} decomposition\n"
            s += f"N_components: {self.n_components}\n"
        else:
            s += f"method: {self.method}\n"
            s += f"N_freq_grid: {self.n_freq_grid}\n"
        s += f"S_low: {self.S_low}\n"
        s += f"S_high: {self.S_high}\n"
        s += self.PSD.__str__()
        return s

    def __repr__(self) -> str:
        return self.__str__()