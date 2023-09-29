"""Generate time series from a given PSD or ACVF model.
"""
import warnings

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from jax import random
from jax.scipy.linalg import cholesky
from numpy import savetxt
from scipy.interpolate import interp1d

from .acvf_base import CovarianceFunction
from .psd_base import PowerSpectralDensity
from .utils import EuclideanDistance


class Simulations:
    """Simulate time series from a given PSD or ACVF model.


    Parameters
    ----------
    T : :obj:`float`
        duration of the time series.
    dt : :obj:`float`
        sampling period of the time series.
    model : :class:`~pioran.acvf_base.CovarianceFunction` or :class:`~pioran.psd_base.PowerSpectralDensity`
        The model for the simulation of the process, can be a PSD or an ACVF.
    S_low : :obj:`float`, optional
        Scale factor for the lower bound of the frequency grid.
        If the model is a PSD, this parameter is mandatory.
    S_high : :obj:`float`, optional
        Scale factor for the upper bound of the frequency grid.
        If the model is a PSD, this parameter is mandatory.

    Raises
    ------
    ValueError
        If the model is not a PSD or ACVF.
    """

    duration: float
    """duration of the time series."""
    sampling_period: float
    """sampling period of the time series."""
    n_time: int
    """number of time indexes."""
    t: jax.Array
    """time :obj:`jax.Array` of the time series."""
    f_max_obs: float
    """maximum frequency of the observed frequency grid."""
    f_min_obs: float
    """minimum frequency of the observed frequency grid."""
    f0: float
    """minimum frequency of the total frequency grid."""
    fN: float
    """maximum frequency of the total frequency grid."""
    n_freq_grid: int
    """number of frequency indexes."""
    frequencies: jax.Array
    """frequency array of the total frequency grid."""
    tau_max: float
    """maximum lag of the autocovariance function."""
    dtau: float
    """sampling period of the autocovariance function."""
    tau: jax.Array
    """lag array of the autocovariance function."""
    psd: jax.Array
    """power spectral density of the time series."""
    acvf: jax.Array
    """autocovariance function of the time series."""
    triang: bool
    """triangular matrix used to generate the time series with the Cholesky decomposition."""
    keys: dict
    """dictionary of the keys used to generate the random numbers. See :func:`~pioran.simulate.Simulations.generate_keys` for more details."""

    def __init__(
        self,
        T,
        dt,
        model: PowerSpectralDensity | CovarianceFunction,
        N=None,
        S_low=None,
        S_high=None,
    ):
        if not isinstance(model, PowerSpectralDensity) and not isinstance(
            model, CovarianceFunction
        ):
            raise ValueError(
                f"You must provide a model which is either a PowerSpectralDensity or a CovarianceFunction, not {type(model)}"
            )

        if isinstance(model, PowerSpectralDensity) and (
            (S_low is None) or (S_high is None)
        ):
            raise ValueError(
                "When the model is a PSD, you must provide S_low and S_high for the frequency grid"
            )
        # case where the model is an ACVF, we dont really care about the frequency grid
        # nor the S_low and S_high parameters, we just need S_low=2 for plotting purposes
        elif isinstance(model, CovarianceFunction) and (
            (S_low is None) or (S_high is None)
        ):
            S_low = 2
            S_high = 1

        self.S_low = S_low
        self.S_high = S_high
        # parameters of the time series
        self.duration = T
        self.sampling_period = dt
        self.n_time = jnp.rint(T / dt).astype(int) if N is None else N
        self.t = jnp.arange(0, self.duration, self.sampling_period)

        # parameters of the **observed** frequency grid
        self.f_max_obs = 0.5 / dt
        self.f_min_obs = 1 / T

        # parameters of the **total** frequency grid
        self.f0 = self.f_min_obs / S_low
        self.fN = self.f_max_obs * S_high
        self.n_freq_grid = jnp.rint(self.fN / self.f0).astype(int) + 1

        self.frequencies = jnp.arange(0, self.fN + self.f0, self.f0)
        self.tau_max = 0.5 / self.f0  # 0.5/self.f0
        self.dtau = 0.5 / self.fN  # self.tau_max/(self.n_freq_grid-1)
        self.tau = jnp.arange(0, self.tau_max + self.dtau, self.dtau)

        self.psd = None
        self.triang = None
        self.acvf = None
        self.keys = {}

        if isinstance(model, PowerSpectralDensity):
            self.psd = model.calculate(self.frequencies[1:])
            self.psd = jnp.insert(self.psd, 0, 0)
        else:
            self.acvf = model.calculate(self.tau)

        self.model = model

    def generate_keys(self, seed: int = 0) -> None:
        """Generate the keys to generate the random numbers.

        This function generates the keys to generate the random numbers for the simulations and store them in the dictionary self.keys.
        The keys and their meaning are:


        - `simu_TS`  : key for drawing the values of the time series.
        - `errors`   : key for drawing the size of the errorbar of the time series from a given distribution.
        - `fluxes`   : key for drawing the fluxes of the time series from a given distribution.
        - `subset`   : key for randomising the choice of the subset of the time series.
        - `sampling` : key for randomising the choice of the sampling of the time series.

        Parameters
        ----------
        seed  : :obj:`int`, optional
            Seed for the random number generator, by default 0
        """

        key = random.PRNGKey(seed)
        (
            self.keys["ts"],
            self.keys["errors"],
            self.keys["fluxes"],
            self.keys["subset"],
            self.keys["sampling"],
        ) = random.split(key, 5)

    def plot_acvf(self, figsize=(9, 5.5), xunit="d", filename=None, title=None):
        """Plot the autocovariance function.

        Plot the autocovariance function of the time series.

        Parameters
        ----------
        figsize  : :obj:`tuple`, optional
            Size of the figure, by default (15,3)
        xunit  : :obj:`str`, optional
            Unit of the x-axis, by default 'd'
        filename  : :obj:`str`, optional
            Name of the file to save the figure, by default None
        title  : :obj:`str`, optional
            Title of the plot, by default None

        Returns
        -------
        :obj:`matplotlib.figure.Figure`
            Figure of the plot
         :obj:`matplotlib.axes.Axes`
            Axes of the plot

        """

        if self.acvf is None:
            raise NotImplementedError(
                "Plotting the ACV when the PSD is modelled is not implemented yet"
            )

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.plot(self.tau, self.acvf, "-")
        ax.set_xlim(0, self.duration)
        ax.set_xlabel(r"Time lag $ \tau" + f"(\mathrm{{{xunit}}})$")
        ax.set_ylabel("Autocovariance function")
        if title is not None:
            ax.set_title(title)
        fig.tight_layout()

        if filename is not None:
            fig.savefig(f"{filename}", format="pdf")
        return fig, ax

    def plot_psd(
        self, figsize=(6, 4), filename=None, title=None, xunit="d", loglog=True
    ):
        """Plot the power spectral density model.

        A plot of the power spectral density model is generated.

        Parameters
        ----------
        figsize  : :obj:`tuple`, optional
            Size of the figure, by default (6,4)
        filename  : :obj:`str`, optional
            Name of the file to save the figure, by default None
        title  : :obj:`str`, optional
            Title of the plot, by default None
        xunit  : :obj:`str`, optional
            Unit of the x-axis, by default 'd'
        loglog  : :obj:`bool`, optional
            If True, the plot is in loglog, by default True

        Returns
        -------
        :obj:`matplotlib.figure.Figure`
            Figure of the plot
        :obj:`matplotlib.axes.Axes`
            Axes of the plot

        """

        if self.psd is None:
            raise NotImplementedError(
                "Plotting the PSD when the model is the ACV is not implemented yet"
            )

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.plot(self.frequencies, self.psd, ".-")
        ax.axvline(self.f_max_obs, label=r"$f_{\rm max}$", color="C1", ls="--")
        ax.axvline(self.f_min_obs, label=r"$f_{\rm min}$", color="C4", ls=":")
        if loglog:
            ax.loglog()
        ax.legend()
        ax.set_xlabel(f"Frequency $(\mathrm{{{xunit}}}^{{-1}})$")
        ax.set_ylabel("PSD")
        if title is not None:
            ax.set_title(title)
        fig.tight_layout()

        if filename is not None:
            fig.savefig(f"{filename}", format="pdf")
        return fig, ax

    def GP_method(
        self, t_test: jax.Array, interpolation="cubic"
    ) -> tuple[jax.Array, jax.Array]:
        """Generate a time series using the GP method.

        If the ACVF is not already calculated, it is calculated from the PSD
        using the inverse Fourier transform.

        Parameters
        ----------
        t_test: :obj:`jax.Array`
            Time array of the time series.
        interpolation  : :obj:`str`, optional
            Interpolation method to use for the GP function, by default 'cubic'.

        Raises
        ------
        ValueError
            If the interpolation method is not 'linear' or 'cubic'.

        Returns
        -------
        :obj:`jax.Array`
            Time array of the time series.
        :obj:`jax.Array`
            Time series.
        """
        # if no acvf is given, we calculate it from the psd
        if self.acvf is None:
            acv = jnp.fft.irfft(self.psd)
            self.acvf = acv[: len(acv) // 2 + 1] / self.dtau

        # if the cholesky decomposition of the autocovariance function is not already calculated, we calculate it
        if self.triang is None:
            dist = EuclideanDistance(t_test.reshape(-1, 1), t_test.reshape(-1, 1))

            if isinstance(self.model, CovarianceFunction):
                K = self.model.calculate(dist)
            else:
                if interpolation == "linear":
                    interpo = interp1d(self.tau, self.acvf, "linear")
                elif interpolation == "cubic":
                    interpo = interp1d(self.tau, self.acvf, "cubic")
                else:
                    raise ValueError(
                        f"Interpolation method {interpolation} is not accepted, use either 'linear' or 'cubic'"
                    )
                K = interpo(dist)

            self.triang = cholesky(K).T

        r = random.normal(key=self.keys["ts"], shape=(len(t_test),))
        ts = self.triang @ r
        return t_test, ts

    def simulate(
        self,
        mean=None,
        method: str = "GP",
        irregular_sampling: bool = False,
        irregular_gaps: bool = False,
        randomise_fluxes: bool = True,
        errors: str = "gauss",
        errors_size: float = 0.02,
        N_points: int = 0,
        seed: int = 0,
        seed_gaps: int = 1,
        filename: str = "",
        exponentiate_ts: bool = False,
        min_n_gaps: int = 0,
        max_n_gaps: int = 10,
        max_size_slices: float = 2.0,
        interp_method: str = "cubic",
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Method to simulate time series using either the GP method or the TK method.

        When using the GP method, the time series is generated using an analytical autocovariance function or a power spectral density.
        If the autocovariance function is not provided, it is calculated from the power spectral density using the inverse Fourier transform
        and interpolated using a linear interpolation to map the autocovariance function on a grid of time lags.

        When using the TK method, the time series is generated using the :func:`~pioran.simulate.Simulations.timmer_Koenig_method` method for a larger duration and then the final time series
        is obtained by taking a subset of the generate time series.

        If irregular_sampling is set to `True`, the time series will be sampled at random irregular time intervals.


        Parameters
        ----------
        mean : :obj:`float`, optional
            Mean of the time series, if None the mean will be set to -2 min(ts)
        method : :obj:`str`, optional
            method to simulate the time series, by default 'GP'
            can be 'TK' which uses Timmer and Koening method
        randomise_fluxes : :obj:`bool`, optional
            If True the fluxes will be randomised.
        errors : :obj:`str`, optional
            If 'gauss' the errors will be drawn from a gaussian distribution
        errors_size : :obj:`float`, optional
            Size of the errors on the time series, by default 0.02 (2%)
        N_points : :obj:`int`, optional
            Number of points to sample when simulating irregular gaps, by default 0
        irregular_sampling : :obj:`bool`, optional
            If True the time series will be sampled at irregular time intervals
        irregular_gaps : :obj:`bool`, optional
            If True the time series will be sampled at irregular time intervals with random gaps
        seed : :obj:`int`, optional
            Set the seed for the RNG
        seed_gaps : :obj:`int`, optional
            Set the seed for the RNG when simulating irregular gaps
        exponentiate_ts: :obj:`bool`, optional
            Exponentiate the time series to produce a lognormal flux distribution.
        filename : :obj:`str`, optional
            Name of the file to save the time series, by default ''
        min_n_gaps : :obj:`int`, optional
            Minimum number of gaps when simulating irregular gaps, by default 2
        max_n_gaps : :obj:`int`, optional
            Maximum number of gaps when simulating irregular gaps, by default 22
        max_size_slices: obj:`float`
            Max size factor, default it 2.
        interp_method : :obj:`str`, optional
            Interpolation method to use when calculating the autocovariance function from the power spectral density, by default 'linear'
        Raises
        ------
        ValueError
            If the method is not 'GP' or 'TK'
        ValueError
            If the errors are not 'gauss' or 'poisson'

        Returns
        -------
        t : :obj:`jax.Array`
            The time indexes of the time series.
        ts : :obj:`jax.Array`
            Values of the simulated time series.
        ts_err : :obj:`jax.Array`
            Errors on the simulated time series

        """
        if irregular_gaps and N_points == 0:
            raise ValueError("You must provide N_points when simulating irregular gaps")

        self.generate_keys(seed=seed)

        if method == "GP":
            if irregular_sampling:
                t_test = jnp.sort(
                    random.uniform(
                        key=self.keys["sampling"],
                        shape=(self.n_time,),
                        minval=0,
                        maxval=self.duration,
                    )
                )
                while not t_test.shape == jnp.unique(t_test).shape:
                    warnings.warn(
                        "The time series is not sampled at unique time intervals, resampling with new seed"
                    )
                    self.keys["sampling"] += 1
                    t_test = jnp.sort(
                        random.uniform(
                            key=self.keys["sampling"],
                            shape=(self.n_time,),
                            minval=0,
                            maxval=self.duration,
                        )
                    )
            else:
                t_test = jnp.arange(0, self.duration, self.sampling_period)
                # raise NotImplementedError("The GP method does not support irregular sampling yet")
            if self.n_time > 8000:
                warnings.warn(
                    "The desired number of point in the simulated time series is quite high"
                )
            t, true_timeseries = self.GP_method(
                interpolation=interp_method, t_test=t_test
            )

        elif method == "TK":
            if self.psd is None:
                raise ValueError(
                    "You must provide a PowerSpectralDensity model to use the TK method"
                )
            # generate a long time series using the TK method
            long_t, long_true_timeseries = self.timmer_Koenig_method()
            # minimal index of the start of the subset of the time series
            start_subset = jnp.rint(self.duration / (0.5 / self.fN)).astype(int)
            subset_t, subset_true_timeseries = self.extract_subset_timeseries(
                long_t, long_true_timeseries, start_subset
            )
            t, true_timeseries = self.sample_timeseries(
                subset_t,
                subset_true_timeseries,
                self.n_time,
                irregular_sampling=irregular_sampling,
            )
            t -= t[0]

        else:
            raise ValueError(
                f"method {method} is not accepted, use either 'GP' or 'TK'"
            )

        if exponentiate_ts:
            true_timeseries = jnp.exp(true_timeseries)
        if randomise_fluxes:
            if errors == "gauss":
                # generate the variance of the errors
                timeseries_error_size = (errors_size*
                    jnp.sqrt(jnp.abs(true_timeseries))
                    * jnp.abs(random.normal(key=self.keys["errors"], shape=(len(t),)))
                )
                # generate the measured time series with the associated fluxes
                observed_timeseries = (
                    true_timeseries
                    + timeseries_error_size
                    * random.normal(key=self.keys["fluxes"], shape=(len(t),))
                )
            elif errors == "poisson":
                raise NotImplementedError("Poisson errors are not implemented yet")
                #### IMPLEMENT POISSON ERRORS
            else:
                raise ValueError(
                    f"Error type {errors} is not accepted, use either 'gauss' or 'poisson'"
                )

        else:
            timeseries_error_size = jnp.zeros_like(t)
            observed_timeseries = true_timeseries


        # set the mean of the time series
        if mean is not None:
            observed_timeseries = (
                observed_timeseries - jnp.mean(observed_timeseries) + mean
            )
        else:
            observed_timeseries = observed_timeseries + 2 * jnp.abs(
                jnp.min(observed_timeseries)
            )

        if filename != '':
            savetxt(
                filename, jnp.vstack([t, observed_timeseries, timeseries_error_size]).T
            )
        if not irregular_gaps:
            if N_points != 0:
                key = jax.random.PRNGKey(seed_gaps)
                t_points = jnp.sort(jax.random.choice(key, t, shape=(N_points,), replace=False))
                indexes_selected = jnp.searchsorted(t, t_points)
                
                return t_points, observed_timeseries[indexes_selected], timeseries_error_size[indexes_selected]

            return t, observed_timeseries, timeseries_error_size
        else:
            t_gappy, indexes_selected = self.simulate_irregular_gaps(
                t,
                seed=seed_gaps,
                N_points=N_points,
                min_n_gaps=min_n_gaps,
                max_n_gaps=max_n_gaps,
                max_size_slices=max_size_slices
            )
            return (
                t_gappy,
                observed_timeseries[indexes_selected],
                timeseries_error_size[indexes_selected],
            )

    def simulate_irregular_gaps(
        self,
        a: jax.Array,
        seed: int,
        N_points: int,
        min_n_gaps: int = 2,
        max_n_gaps: int = 22,
        max_size_slices: float = 2.0,
    ):
        """Simulate irregular times from a regular time series with random gaps.

        Parameters
        ----------
        a : :obj:`jax.Array`
            Regular time series indexes.
        key : :obj:`jax.random.PRNGKey`
            Random key.
        N_points : :obj:`int`
            Number of points to sample.
        min_n_gaps : :obj:`int`
            Minimum number of gaps. Default is 2.
        max_n_gaps : :obj:`int`
            Maximum number of gaps. Default is 22.
        max_size_slices : obj:`float`
            Max size factor, default it 2.
                
        Returns
        -------
        :obj:`jax.Array`
            Irregular times.
        :obj:`jax.Array`
            Indexes of the irregular times.
        """
        N_tot = len(a)
        key = jax.random.PRNGKey(seed)
        keys = jax.random.split(key, 4)
        sucess = False

        while not sucess:
            N_slices = (
                jax.random.randint(
                    keys[0], minval=min_n_gaps, maxval=max_n_gaps, shape=(1,)
                )
                .squeeze()
                .astype(int)
            )  # number of slices
            min_size_slices = jnp.rint(0.05 * N_tot).astype(int)
            max_size_slices = jnp.rint(1 / (N_slices*max_size_slices) * N_tot).astype(int)
            size_slices = jax.random.randint(
                keys[2],
                minval=min_size_slices,
                maxval=max_size_slices,
                shape=(N_slices,),
            )  # size of each slice

            keys_list = [keys[1]]
            N_start = []
            intervals = [[0, max_size_slices]]

            for i, slice in enumerate(size_slices):
                key, subkey = jax.random.split(keys_list[-1], 2)
                N_start_slices = jax.random.randint(
                    subkey, minval=intervals[i][0], maxval=intervals[i][1], shape=(1,)
                )  # start time of each slice
                keys_list.append(subkey)
                N_start.append(N_start_slices)
                if i == N_slices - 1:
                    # print('Here we can take any startpoint as long as it does not exceed the total number of points')
                    intervals.append(
                        [
                            N_start_slices + slice,
                            N_tot - slice - max_size_slices // 2 - 5,
                        ]
                    )
                    # print(f'Next time draw between {intervals[-1][0]} and {intervals[-1][1]}')
                else:
                    intervals.append(
                        [
                            N_start_slices + slice,
                            N_start_slices + slice + max_size_slices - 5,
                        ]
                    )

            N_start = jnp.array(N_start).flatten()  # start times of each slice
            N_stop = N_start + size_slices  # stop times of each slice
            if jnp.any(N_stop > N_tot):
                sucess = False
                keys = jax.random.split(keys[-1], 4)
                # print("N_stop > N_tot")
                continue
            else:
                sucess = True

            L = jnp.array([], dtype=int)
            for k in range(N_slices):
                L = jnp.append(L, jnp.arange(N_start[k], N_stop[k], 1, dtype=int))

            x_indexes = jnp.arange(0, N_tot, 1)
            irreg = jnp.delete(x_indexes, L)
            trunc = a[irreg]
            if jnp.any(len(trunc) < N_points):
                sucess = False
                keys = jax.random.split(keys[-1], 4)
                # print("N_stop > N_tot")
                continue
            else:
                sucess = True

            t_points = jnp.sort(
                jax.random.choice(keys[3], trunc, shape=(N_points,), replace=False)
            )
            indexes_selected = jnp.searchsorted(a, t_points)
        return t_points, indexes_selected

    def extract_subset_timeseries(
        self, t: jax.Array, y: jax.Array, M: int
    ) -> tuple[jax.Array, jax.Array]:
        """Select a random subset of points from an input time series.

        The input time series is regularly sampled of size N.
        The output time series is of size M with the same sampling rate as the input time series.

        Parameters
        ----------
        t : :obj:`jax.Array`
            Input time series of size N.
        y : :obj:`jax.Array`
            The fluxes of the simulated light curve.
        M : :obj:`int`
            The number of points in the desired time series.

        Returns
        -------
        :obj:`jax.Array`
            The time series of size M.
        :obj:`jax.Array`
            The values of the time series of size M.
        """
        N = len(t)
        start_index = random.randint(
            key=self.keys["subset"], shape=(1,), minval=M, maxval=N - M - 1
        )[0]
        return t[start_index : start_index + M], y[start_index : start_index + M]

    def sample_timeseries(
        self, t: jax.Array, y: jax.Array, M: int, irregular_sampling: bool = False
    ):
        """Extract a random subset of points from the time series.

        Extract a random subset of M points from the time series. The input time series t is regularly sampled of size N with a sampling period dT.
        If irregular_sampling is False, the output time series has a sampling period dT/M.
        If irregular_sampling is True, the output time series is irregularly sampled.

        Parameters
        ----------
        t : :obj:`jax.Array`
            The time indexes of the time series.
        y : :obj:`jax.Array`
            The values of the time series.
        M : :obj:`int`
            The number of points in the desired time series.
        irregular_sampling : :obj:`bool`
            If True, the time series is irregularly sampled. If False, the time series is regularly sampled.

        Returns
        -------
        :obj:`jax.Array`
            The time indexes of the sampled time series.
        :obj:`jax.Array`
            The values of the sampled time series.
        """
        input_sampling_period = t[1] - t[0]
        output_sampling_period = (t[-1] - t[0]) / M

        if not irregular_sampling:
            index_step_size = jnp.rint(
                output_sampling_period / input_sampling_period
            ).astype(int)
            return t[::index_step_size], y[::index_step_size]
        else:
            t_sampled = random.choice(
                key=self.keys["sampling"], a=t, shape=(M,), replace=False
            )
            y_sampled = random.choice(
                key=self.keys["sampling"], a=y, shape=(M,), replace=False
            )
            sorted_indices = jnp.argsort(t_sampled)

            return t_sampled[sorted_indices], y_sampled[sorted_indices]

    def timmer_Koenig_method(self) -> tuple[jax.Array, jax.Array]:
        r"""Generate a time series using the Timmer-Konig method.

        Use the Timmer-Konig method to generate a time series with a given power spectral density
        stored in the attribute psd.

        Assuming a power-law shaped PSD, the method is as follows:

        Draw two independent Gaussian random variables N1 and N2 with zero mean and unit variance.
        The random variables are drawn using the key self.keys['ts'] split into two subkeys.

            1. Define A = sqrt(PSD/2) * (N1 + i*N2)
            2. Define A[0] = 0
            3. Define A[-1] = real(A[-1])
            4. ts = irfft(A)
            5. t is defined as the time indexes of the time series, with a sampling period of 0.5/fN.
            6. ts is multiplied by the 2*len(psd)*sqrt(f0) factor to ensure that the time series has the correct variance.

        The duration of the output time series is 2*(len(psd)-1).

        Returns
        -------
        :obj:`jax.Array`
            The time indexes of the time series.
        :obj:`jax.Array`
            The values of the time series.
        """
        # split the key into two subkeys
        key, subkey = random.split(self.keys["ts"])

        N1, N2 = random.normal(key=key, shape=(len(self.psd),)), random.normal(
            key=subkey, shape=(len(self.psd),)
        )
        randpsd = jnp.sqrt(0.5 * self.psd) * (N1 + 1j * N2)
        randpsd.at[0].set(0)
        randpsd.at[-1].set(jnp.real(randpsd[-1]))

        ts = jnp.fft.irfft(randpsd)
        t = jnp.arange(0, self.dtau * len(ts), self.dtau)
        self.variance = jnp.sum(self.psd * self.f0)
        ts = 2 * ts * jnp.sqrt(self.f0) * len(self.psd)
        return t, ts

    def split_longtimeseries(
        self, t: jax.Array, ts: jax.Array, n_slices: int
    ) -> tuple[list, list]:
        """Split a long time series into shorter time series.

        Break the time series into n_slices shorter time series. The short time series are of equal length.

        Parameters
        ----------
        t : :obj:`jax.Array`
            The time indexes of the long time series.
        ts : :obj:`jax.Array`
            The values of the long time series.
        n_slices : :obj:`int`
            The number of slices to break the time series into.

        Returns
        -------
        :obj:`list`
            A list of the time indexes of the shorter time series.
        :obj:`list`
            A list of the values of the shorter time series.

        """
        t_slices = []
        ts_slices = []
        size_slice = int(len(t) / n_slices)
        for i in range(n_slices):
            start_index = i * size_slice
            end_index = (i + 1) * size_slice
            t_slice, ts_slice = t[start_index:end_index], ts[start_index:end_index]

            t_slices.append(t_slice)
            ts_slices.append(ts_slice)
        return t_slices, ts_slices

    def resample_longtimeseries(
        self, t_slices: list, ts_slices: list
    ) -> tuple[list, list]:
        """Resample the time series to have a regular sampling period with n_time points.

        Parameters
        ----------
        t_slices : :obj:`list`
            A list of short time series time indexes.
        ts_slices : :obj:`list`
            A list of short time series values.

        Returns
        -------
        :obj:`list`
            A list of the time indexes of the sampled time series.
        :obj:`list`
            A list of the values of the sampled time series.

        """
        t_resampled, ts_resampled = [], []

        for t_slice, ts_slice in zip(t_slices, ts_slices):
            input_sampling_period = t_slice[1] - t_slice[0]
            output_sampling_period = (t_slice[-1] - t_slice[0]) / self.n_time

            index_step_size = jnp.rint(
                output_sampling_period / input_sampling_period
            ).astype(int)
            t_resampled.append(t_slice[::index_step_size])
            ts_resampled.append(ts_slice[::index_step_size])
        return t_resampled, ts_resampled

    def simulate_longtimeseries(
        self,
        mean: float | None = None,
        randomise_fluxes: bool = True,
        errors: str = "gauss",
        seed: int = 0,
    ) -> tuple[list, list, list]:
        """Method to simulate several long time series using the Timmer-Koenig method.

        The time series is generated using the :func:`~pioran.simulate.Simulations.timmer_Koenig_method` method for a larger duration and then the final time series
        are split into segments of length n_time. The shorter time series are then resampled to have a regular sampling period.


        Parameters
        ----------
        mean : :obj:`float`, optional
            Mean of the time series, if None the mean will be set to -2 min(ts)
        randomise_fluxes : :obj:`bool`, optional
            If True the fluxes will be randomised.
        errors : :obj:`str`, optional
            If 'gauss' the errors will be drawn from a gaussian distribution
        seed : :obj:`int`, optional
            Set the seed for the RNG

        Raises
        ------
        ValueError
            If the errors are not 'gauss' or 'poisson'

        Returns
        -------
        :obj:`list`
            A list of the time indexes of the segments.
        :obj:`list`
            A list of the values of the segments.
        :obj:`list`
            A list of the errors of the segments.

        """
        self.generate_keys(seed=seed)

        if self.psd is None:
            raise ValueError(
                "You must provide a PowerSpectralDensity model to use the TK method"
            )
        # generate a long time series using the TK method
        long_t, long_true_timeseries = self.timmer_Koenig_method()

        # split the long time series into segments given by the S_low parameter
        if not isinstance(self.S_low, int):
            warnings.warn(
                "The number of slices is not an integer, it will be rounded to the nearest integer"
            )
        n_slices = int(self.S_low)
        if n_slices < 10:
            warnings.warn(
                "The number of slices is less than 10, this may cause leakage between the segments"
            )
        t_slices, ts_slices = self.split_longtimeseries(
            long_t, long_true_timeseries, n_slices
        )

        # resample the segments to have a regular sampling period
        t_resampled, ts_resampled = self.resample_longtimeseries(t_slices, ts_slices)
        # set origin of time to 0 for each segment
        for i in range(len(t_resampled)):
            t_resampled[i] = t_resampled[i] - t_resampled[i][0]

        ts_err = []
        ts = []
        if randomise_fluxes:
            # split the keys for the fluxes and errors for each segment
            keys_errors = random.split(self.keys["errors"], n_slices)
            keys_fluxes = random.split(self.keys["fluxes"], n_slices)

            if errors == "gauss":
                for i in range(n_slices):
                    # generate the variance of the errors
                    timeseries_error_size = jnp.abs(
                        random.normal(key=keys_errors[i], shape=(len(t_resampled[i]),))
                    )
                    # generate the measured time series with the associated fluxes
                    observed_timeseries = ts_resampled[
                        i
                    ] + timeseries_error_size * random.normal(
                        key=keys_fluxes[i], shape=(len(t_resampled[i]),)
                    )

                    ts.append(observed_timeseries)
                    ts_err.append(timeseries_error_size)

            elif errors == "poisson":
                raise NotImplementedError("Poisson errors are not implemented yet")
                #### IMPLEMENT POISSON ERRORS
            else:
                raise ValueError(
                    f"Error type {errors} is not accepted, use either 'gauss' or 'poisson'"
                )

        else:
            for i in range(n_slices):
                ts_err.append(jnp.zeros_like(t_resampled[i]))
                ts.append(ts_resampled[i])

        if mean is not None:
            for i in range(n_slices):
                ts[i] = ts[i] - jnp.mean(ts[i]) + mean
        else:
            for i in range(n_slices):
                ts[i] = ts[i] - 2 * jnp.min(ts[i])

        return t_resampled, ts, ts_err
