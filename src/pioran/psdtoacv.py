"""Convert a power spectral density to an autocovariance function via the inverse Fourier transform methods or kernel decomposition.
"""
import equinox as eqx
import jax
import jax.numpy as jnp
import tinygp

try:
    from celerite2.jax import terms
    import celerite2.terms as legacy_terms
except ImportError:
    terms = None
    legacy_terms = None

from tinygp.kernels.quasisep import SHO as SHO_term

from .parameters import ParametersModel
from .psd_base import PowerSpectralDensity
from .utils import (
    EuclideanDistance,
    decompose_triangular_matrix,
    reconstruct_triangular_matrix,
    valid_methods,
)

_USE_JAX_FINUFFT = True
try:
    from jax_finufft import nufft2
except ImportError:
    nufft2 = None
    _USE_JAX_FINUFFT = False


class PSDToACV(eqx.Module):
    """Represents the tranformation of a power spectral density to an autocovariance function.

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
    init_variance : :obj:`float`, optional
        Initial value of the variance. Default is 1.0.

    Raises
    ------
    TypeError
        If PSD is not a :class:`~pioran.psd_base.PowerSpectralDensity` object.
    ValueError
        If S_low is smaller than 2., if method is not in the allowed methods or if n_components is smaller than 1.
    """

    PSD: PowerSpectralDensity
    """Power spectral density object."""
    ACVF: tinygp.kernels.quasisep.SHO
    """Autocovariance function as sum of SHO kernels."""
    parameters: ParametersModel
    """Parameters of the power spectral density."""
    method: str
    """Method to compute the covariance function from the power spectral density, by default 'FFT'.Possible values are:
            - 'FFT': use the FFT to compute the autocovariance function.
            - 'NuFFT': use the non-uniform FFT to compute the autocovariance function.
            - 'SHO': approximate the power spectrum as a sum of SHO basis functions to compute the autocovariance function.
            - 'DRWCelerite' : approximate the power spectrum as a sum of DRW+Celerite basis functions to compute the autocovariance function."""
    f_max_obs: float
    """Maximum observed frequency, i.e. the Nyquist frequency."""
    f_min_obs: float
    """Minimum observed frequency."""
    f0: float
    """Lower bound of the frequency grid."""
    S_low: float
    """Scale for the lower bound of the frequency grid."""
    S_high: float
    """Scale for the upper bound of the frequency grid."""
    fN: float
    """Upper bound of the frequency grid."""
    estimate_variance: bool
    """If True, the amplitude of the autocovariance function is estimated."""
    n_freq_grid: int | None = None
    """Number of points in the frequency grid."""
    frequencies: jax.Array | None = None
    """Frequency grid."""
    tau: jax.Array = 0
    """Time lag grid."""
    dtau: float = 0
    """Time lag step."""
    n_components: int = 0
    """Number of components used to approximate the power spectral density using the 'SHO' method."""
    spectral_points: jax.Array | None = None
    """Frequencies of the SHO kernels."""
    spectral_matrix: jax.Array | None = None
    """Matrix of the SHO kernels."""
    use_celerite: bool = False
    """Use celerite2 as a backend to model the autocovariance function and compute the log marginal likelihood."""
    use_legacy_celerite: bool = False
    """Use celerite2 as a backend to model the autocovariance function and compute the log marginal likelihood."""

    def __init__(
        self,
        PSD: PowerSpectralDensity,
        S_low: float,
        S_high: float,
        T: float,
        dt: float,
        method: str,
        n_components: int = 0,
        estimate_variance: bool = True,
        init_variance: float = 1.0,
        use_celerite=False,
        use_legacy_celerite: bool = False,
    ):
        """Constructor of the PSDToACV class."""

        self.estimate_variance = estimate_variance

        # sanity checks:
        if not isinstance(PSD, PowerSpectralDensity):
            raise TypeError(
                f"PSD must be a PowerSpectralDensity object, not a {type(PSD)}"
            )

        if dt > T:
            raise ValueError(f"dt ({dt}) must be smaller than T ({T})")

        if S_low < 2:
            raise ValueError(f"S_low must be greater than 2, {S_low} was given")

        if method not in valid_methods:
            raise ValueError(
                f"Method {method} not allowed. Choose between {valid_methods}"
            )
        if method == "NuFFT" and (not _USE_JAX_FINUFFT):
            raise ValueError("NuFFT method requires jax_finufft package")

        if ("FFT" not in method) and n_components < 1:
            raise ValueError("n_components must be greater than 1")

        # define the attributes
        self.PSD = PSD
        self.parameters = PSD.parameters
        self.method = method
        self.S_low = S_low
        self.S_high = S_high
        self.use_celerite = use_celerite
        self.use_legacy_celerite = use_legacy_celerite

        if self.estimate_variance:
            self.parameters.append("var", init_variance, True, hyperparameter=False)

        # parameters of the **observed** frequency grid
        self.f_max_obs = 0.5 / dt  # Nyquist frequency
        self.f_min_obs = 1 / T  # minimum observed frequency

        # parameters of the **total** frequency grid
        self.f0 = self.f_min_obs / self.S_low  # minimum frequency
        self.fN = self.f_max_obs * self.S_high  # maximum frequency

        self.n_freq_grid = jnp.rint(jnp.ceil(self.fN / self.f0)) + 1

        tau_max = 0.5 / self.f0
        self.dtau = tau_max / (self.n_freq_grid - 1)

        if self.method == "FFT" or self.method == "NuFFT":
            self.frequencies = jnp.arange(0, self.fN + self.f0, self.f0)
            self.tau = jnp.arange(0, tau_max + self.dtau, self.dtau)

        # here we define the spectral matrix for the PSD approximation
        elif self.method == "SHO":
            self.n_components = n_components
            self.spectral_points = jnp.geomspace(self.f0, self.fN, self.n_components)
            self.spectral_matrix = 1 / (
                1
                + jnp.power(
                    jnp.atleast_2d(self.spectral_points).T / self.spectral_points, 4
                )
            )
        elif self.method == "DRWCelerite":
            self.n_components = n_components
            self.spectral_points = jnp.geomspace(self.f0, self.fN, self.n_components)
            self.spectral_matrix = 1 / (
                1
                + jnp.power(
                    jnp.atleast_2d(self.spectral_points).T / self.spectral_points, 6
                )
            )
        else:
            raise NotImplementedError(f"Method {self.method} not implemented")

    def decompose_model(self, psd_normalised: jax.Array):
        r"""Decompose the PSD model into a sum of basis functions.

        Assuming that the PSD model can be written as a sum of :math:`J` , this method
        solve the linear system to find the amplitude :math:`a_j` of each kernel.

        .. math:: :label: sho_power_spectrum

        \boldsymbol{y} = B \boldsymbol{a}

        with :math:`\boldsymbol{y}=\begin{bmatrix}1 & \mathcal{P}(f_1)/\mathcal{P}(f_0) & \cdots & \mathcal{P}(f_J)/\mathcal{P}(f_0) \end{bmatrix}^\mathrm{T}`
        the normalised power spectral density vector, :math:`B` the spectral matrix associated to the linear system and :math:`\boldsymbol{a}` the amplitudes of the functions.


        .. math:: :label: sho_spectral_matrix

        B_{ij} = \dfrac{1}{1 + \left(\dfrac{f_i}{f_j}\right)^4}


        .. math:: :label: drwcel_spectral_matrix

        B_{ij} = \dfrac{1}{1 + \left(\dfrac{f_i}{f_j}\right)^6}


        Parameters
        ----------
        psd_normalised : :obj:`jax.Array`
            Normalised power spectral density by the first value.

        Returns
        -------
        :obj:`jax.Array`
            Amplitudes of the functions.
        :obj:`jax.Array`
            Frequencies of the function.
        """

        a = jnp.linalg.solve(self.spectral_matrix, psd_normalised)
        return a, self.spectral_points

    def get_approx_coefs(self):
        """Get the amplitudes and frequencies of the basis functions.

        Estimate the amplitudes and frequencies of the basis functions by solving the linear system.

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

    def build_SHO_model_legacy_cel(
        self, amplitudes: jax.Array, frequencies: jax.Array
    ) :#-> terms.Term:
        """Build the semi-separable SHO model in celerite from the amplitudes and frequencies.

        Currently multiplying the amplitudes to the SHO kernels as sometimes we need negative amplitudes.
        The amplitudes are modelled as a DRW model with c=0.

        Parameters
        ----------
        amplitudes : :obj:`jax.Array`
            Amplitudes of the SHO kernels.
        frequencies : :obj:`jax.Array`
            Frequencies of the SHO kernels.

        Returns
        -------
        :obj:`term.Term`
            Constructed SHO kernel.
        """
        kernel = legacy_terms.RealTerm(a=amplitudes[0], c=0) * legacy_terms.SHOTerm(
            sigma=1, Q=1 / jnp.sqrt(2), w0=2 * jnp.pi * frequencies[0]
        )
        for j in range(1, self.n_components):
            kernel += legacy_terms.RealTerm(
                a=amplitudes[j], c=0
            ) * legacy_terms.SHOTerm(
                sigma=1, Q=1 / jnp.sqrt(2), w0=2 * jnp.pi * frequencies[j]
            )
        return kernel

    def build_SHO_model_cel(
        self, amplitudes: jax.Array, frequencies: jax.Array
    ) :#-> terms.Term:
        """Build the semi-separable SHO model in celerite from the amplitudes and frequencies.

        Currently multiplying the amplitudes to the SHO kernels as sometimes we need negative amplitudes.
        The amplitudes are modelled as a DRW model with c=0.

        Parameters
        ----------
        amplitudes : :obj:`jax.Array`
            Amplitudes of the SHO kernels.
        frequencies : :obj:`jax.Array`
            Frequencies of the SHO kernels.

        Returns
        -------
        :obj:`term.Term`
            Constructed SHO kernel.
        """
        kernel = terms.RealTerm(a=amplitudes[0], c=0) * terms.SHOTerm(
            sigma=1, Q=1 / jnp.sqrt(2), w0=2 * jnp.pi * frequencies[0]
        )
        for j in range(1, self.n_components):
            kernel += terms.RealTerm(a=amplitudes[j], c=0) * terms.SHOTerm(
                sigma=1, Q=1 / jnp.sqrt(2), w0=2 * jnp.pi * frequencies[j]
            )
        return kernel

    def build_DRWCelerite_model_cel(
        self, amplitudes: jax.Array, frequencies: jax.Array
    )  :#-> terms.Term:
        """Build the semi-separable DRW+Celerite model in celerite from the amplitudes and frequencies.

        The amplitudes

        Parameters
        ----------
        amplitudes : :obj:`jax.Array`
            Amplitudes of the SHO kernels.
        frequencies : :obj:`jax.Array`
            Frequencies of the SHO kernels.

        Returns
        -------
        :obj:`term.Term`
            Constructed SHO kernel.
        """
        w_j = 2 * jnp.pi * frequencies
        a = amplitudes * w_j / 6
        b = jnp.sqrt(3) * w_j * amplitudes / 6
        c = w_j / 2
        d = w_j * jnp.sqrt(3) / 2
        kernel = terms.RealTerm(a=a[0], c=w_j[0]) + terms.ComplexTerm(
            a=a[0], b=b[0], c=c[0], d=d[0]
        )
        for j in range(1, self.n_components):
            kernel += terms.RealTerm(a=a[j], c=w_j[j]) + terms.ComplexTerm(
                a=a[j], b=b[j], c=c[j], d=d[j]
            )
        return kernel

    def build_SHO_model_tinygp(
        self, amplitudes: jax.Array, frequencies: jax.Array
    ) -> tinygp.kernels.quasisep.SHO:
        """Build the semi-separable SHO model in tinygp from the amplitudes and frequencies.

        Currently multiplying the amplitudes to the SHO kernels as sometimes we need negative amplitudes,
        which is not possible with the SHO kernel implementation in tinygp.

        Parameters
        ----------
        amplitudes : :obj:`jax.Array`
            Amplitudes of the SHO kernels.
        frequencies : :obj:`jax.Array`
            Frequencies of the SHO kernels.

        Returns
        -------
        :obj:`tinygp.kernels.quasisep.SHO`
            Constructed SHO kernel.
        """

        kernel = amplitudes[0] * SHO_term(
            quality=1 / jnp.sqrt(2), omega=2 * jnp.pi * frequencies[0]
        )
        for j in range(1, self.n_components):
            kernel += amplitudes[j] * SHO_term(
                quality=1 / jnp.sqrt(2), omega=2 * jnp.pi * frequencies[j]
            )
        return kernel

    @property
    def ACVF(self) -> tinygp.kernels.quasisep.SHO:
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
        if self.method == "SHO":
            if self.use_celerite:
                if self.use_legacy_celerite:
                    kernel = self.build_SHO_model_legacy_cel(a * f, f)
                else:
                    kernel = self.build_SHO_model_cel(a * f, f)
            else:
                kernel = self.build_SHO_model_tinygp(a * f, f)
        elif self.method == "DRWSHO":
            if self.use_celerite:
                if self.use_legacy_celerite:
                    raise NotImplementedError("Not implemented for legacy celerite")
                else:
                    kernel = self.build_DRWCelerite_model_cel(a, f)
            else:
                raise NotImplementedError(
                    "DRWCelerite is only implemented for the celerite2 backend"
                )
        else:
            raise NotImplementedError("Only SHO is implemented for now")

        if self.estimate_variance:
            if self.use_celerite:
                if self.method == "SHO":
                    if self.use_legacy_celerite:
                        return (
                            legacy_terms.RealTerm(
                                a=self.parameters["var"].value / jnp.sum(a * f), c=0
                            )
                            * kernel
                        )
                    return (
                        terms.RealTerm(
                            a=self.parameters["var"].value / jnp.sum(a * f), c=0
                        )
                        * kernel
                    )
                elif self.method == "DRWSHO":
                    if self.use_legacy_celerite:
                        raise NotImplementedError("Not implemented for legacy celerite")
                    return (
                        terms.RealTerm(
                            a=self.parameters["var"].value
                            / jnp.sum(a * f * 2 * jnp.pi / 3),
                            c=0,
                        )
                        * kernel
                    )

                else:
                    raise NotImplementedError(
                        "The estimation of the variance is implemented for SHO and DRWSHO only"
                    )

            return kernel * (self.parameters["var"].value / jnp.sum(a * f))
        return kernel

    def calculate(self, t: jax.Array, with_ACVF_factor: bool = False) -> jax.Array:
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
        if self.method == "FFT":
            psd = self.PSD.calculate(self.frequencies[1:])
            # add a zero at the beginning to account for the zero frequency
            psd = jnp.insert(psd, 0, 0)
            acvf = self.get_acvf_byFFT(psd)

            if self.estimate_variance:
                # normalize by the variance instead of integrating the PSD with the trapezium rule
                R = acvf / acvf[0]
                if not with_ACVF_factor:
                    return self.interpolation(t, R) * self.parameters["var"].value
                else:
                    return (
                        self.interpolation(t, R) * self.parameters["var"].value,
                        acvf[0],
                    )
            return self.interpolation(t, acvf)

        elif self.method == "NuFFT":
            if self.estimate_variance:
                raise NotImplementedError("estimate_variance not implemented for NuFFT")

            N = 2 * (self.n_freq_grid - 1)
            k = jnp.arange(-N / 2, N / 2) * self.f0
            psd = self.PSD.calculate(k) + 0j
            return self.get_acvf_byNuFFT(psd, t * 4 * jnp.pi**2)

        else:
            raise NotImplementedError(f"Method {self.method} not implemented")

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
        return nufft2(psd, t / P).real * self.f0

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
        acvf = acvf[: len(self.tau)] / self.dtau
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
        xq : :obj:`jax.Array`
            First array.
        xp : :obj:`jax.Array`
            Second array.

        Raises
        ------
        NotImplementedError
            If the method is not implemented.

        Returns
        -------
        :obj:`jax.Array`
            Covariance matrix.
        """
        # Compute the Euclidean distance between the query and the points
        dist = EuclideanDistance(xq, xp)

        if self.method == "NuFFT":
            if xq.shape == xp.shape:
                unique, reverse_indexes, triu_indexes, n = decompose_triangular_matrix(
                    dist
                )
                avcf_unique = self.calculate(unique)
                return reconstruct_triangular_matrix(
                    avcf_unique, reverse_indexes, triu_indexes, n
                )
            else:
                d = dist.flatten()
                return self.calculate(d).reshape(dist.shape)
        elif self.method == "FFT":
            # Compute the covariance matrix
            return self.calculate(dist)
        else:
            raise NotImplementedError(
                f"Calculating the covariance matrix for method '{self.method}' not implemented"
            )

    def __str__(self) -> str:
        """String representation of the PSDToACV object.

        Returns
        -------
        :obj:`str`
            String representation of the PSDToACV object.
        """
        s = f"PSDToACV\n"
        if self.method != "NuFFT" and self.method != "FFT":
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
        """Representation of the PSDToACV object."""
        return self.__str__()
