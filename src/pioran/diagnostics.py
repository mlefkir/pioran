import os
import sys
from copy import deepcopy

import jax
import jax.numpy as jnp
import numpy as np

from .carma.carma_core import CARMAProcess
from .carma.carma_utils import (
    CARMA_autocovariance,
    CARMA_powerspectrum,
    MA_quad_to_coeff,
    quad_to_coeff,
)
from .core import GaussianProcess
from .plots import (
    plot_posterior_predictive_ACF,
    plot_posterior_predictive_PSD,
    plot_prediction,
    plot_residuals,
)
from .psdtoacv import PSDToACV
from .utils import SHO_power_spectrum, DRWCelerite_power_spectrum

import matplotlib.pyplot as plt


class Visualisations:
    """Class for visualising the results after an inference run.

    Parameters
    ----------
    process : :obj:`GaussianProcess` or :obj:`CARMAProcess`
        The process to be visualised.
    filename : :obj:`str`
        The filename prefix for the output plots.
    n_frequencies : :obj:`int`, optional
        The number of frequencies at which to evaluate the PSDs, by default 2500.
    """

    process: GaussianProcess | CARMAProcess
    """The process to be visualised."""
    x: jax.Array
    """The observation times."""
    y: jax.Array
    """The observation values."""
    yerr: jax.Array
    """The observation errors."""
    predictive_mean: jax.Array
    """The predictive mean."""
    predictive_cov: jax.Array
    """The predictive covariance."""
    x_pred: jax.Array
    """The prediction times."""
    f_min: float
    """The minimum frequency."""
    f_max: float
    """The maximum frequency."""
    frequencies: jax.Array
    """The frequencies at which to evaluate the PSDs."""
    tau: jax.Array
    """The times at which to evaluate the ACFs."""
    filename_prefix: str
    """The filename prefix for the output plots."""
    process_legacy: GaussianProcess

    def __init__(
        self,
        process: GaussianProcess | CARMAProcess,
        filename: str,
        n_frequencies: int = 2500,
    ) -> None:
        """Initialise the class."""
        self.process = process

        self.x = process.observation_indexes.flatten()
        self.y = process.observation_values.flatten()
        self.yerr = process.observation_errors.flatten()

        self.f_min = 1 / (self.x[-1] - self.x[0])
        self.f_max = 0.5 / jnp.min(jnp.diff(self.x))
        self.frequencies = jnp.logspace(
            jnp.log10(self.f_min), jnp.log10(self.f_max), n_frequencies
        )
        self.tau = jnp.linspace(0, self.x[-1], 1000)
        self.filename_prefix = filename

        if isinstance(self.process, GaussianProcess):
            if self.process.use_legacy_celerite:
                self.process_legacy = self.process
            else:
                psd = deepcopy(self.process.model.PSD)
                self.process_legacy = GaussianProcess(
                    psd,
                    process.observation_indexes,
                    process.observation_values,
                    process.observation_errors,
                    S_low=process.model.S_low,
                    S_high=process.model.S_high,
                    method=process.model.method,
                    estimate_variance=process.model.estimate_variance,
                    estimate_mean=process.estimate_mean,
                    log_transform=process.log_transform,
                    scale_errors=process.scale_errors,
                    n_components=process.model.n_components,
                    use_tinygp=False,
                    use_celerite=True,
                    use_legacy_celerite=True,
                )

    def plot_timeseries_diagnostics(
        self, samples, prediction_indexes: jax.Array | None = None, n_samples: int = 400
    ) -> None:
        """Plot the timeseries diagnostics using samples from the posterior distribution.

        This function will call the :func:`plot_prediction` and :func:`plot_residuals` functions to
        plot the predicted timeseries and the residuals.

        Parameters
        ----------
        samples : :obj:`jax.Array`
            The samples from the posterior distribution.
        prediction_indexes : :obj:`jax.Array`, optional
            The prediction times, by default None
        n_samples : :obj:`int`, optional
            The number of samples to use for the posterior predictive checks, by default 400
        **kwargs
            Additional keyword arguments to be passed to the :func:`plot_prediction` and :func:`plot_residuals` functions.
        """
        print("Plotting timeseries diagnostics...")
        log_transform = (
            False
            if (
                (
                    not (
                        self.process_legacy.use_tinygp
                        or self.process_legacy.use_celerite
                    )
                )
                and self.process_legacy.log_transform
            )
            else None
        )

        print("Sampling from the posterior residuals...")
        t_cont = jnp.unique(
            jnp.concatenate((jnp.linspace(self.x[0], self.x[-1], 1000), self.x))
        )
        indexes_obs = np.searchsorted(t_cont, self.x)
        n_points = len(t_cont)
        conditioned_realisations = np.zeros((n_points, n_samples))

        if not os.path.isfile(f"{self.filename_prefix}conditioned_realisations.txt"):
            for j in range(n_samples):
                print(f"{j}/{n_samples}", end="\r")
                sys.stdout.flush()
                # change the values of the parameters
                self.process.model.parameters.set_free_values(samples[j, :])
                # get the celerite GP
                gp_cel_legacy = self.process.build_gp_celerite_legacy()

                # latent values
                y_cel = jnp.log(
                    jnp.abs(
                        self.process.observation_values
                        - self.process.model.parameters["const"].value
                    )
                )
                # condition the GP
                gp_cond = gp_cel_legacy.condition(y_cel, t_cont)
                # sample from the GP and transform back to the original space
                conditioned_realisations[:, j] = jnp.exp(gp_cond.sample())
            np.savetxt(
                f"{self.filename_prefix}conditioned_realisations.txt",
                conditioned_realisations,
            )
        else:
            conditioned_realisations = np.loadtxt(
                f"{self.filename_prefix}conditioned_realisations.txt"
            )

        # plot the posterior predictive checks for the residuals
        res = (
            self.y.T - conditioned_realisations[indexes_obs, :n_samples].T
        ) / self.yerr.T
        res_bands = np.percentile(res, [2.5, 16, 50, 84, 97.5], axis=0)
        res_mean = np.mean(res, axis=0)

        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.plot(
            self.x,
            res_bands[2],
            label="median",
            marker=".",
            linestyle="none",
            color="b",
        )
        ax.plot(self.x, res_mean, label="mean", marker=".", linestyle="none", color="k")
        ax.fill_between(self.x, res_bands[1], res_bands[3], alpha=0.5, label="68%")
        ax.fill_between(self.x, res_bands[0], res_bands[4], alpha=0.2, label="95%")
        ax.update(
            {"xlabel": "Time (days)", "ylabel": "Residuals/Error", "title": "Residuals"}
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(f"{self.filename_prefix}_residuals.png", dpi=300)


        ## plot the posterior predictive checks for the timeseries
        realisations_percentiles = np.percentile(
            conditioned_realisations, [2.5, 16, 50, 84, 97.5], axis=1
        )
        realisations_mean = np.mean(conditioned_realisations, axis=1)

        fig, ax = plt.subplots(1, 1, figsize=(15, 6.5))

        ax.plot(t_cont, realisations_percentiles[2], label="median", color="b")
        ax.plot(t_cont, realisations_mean, label="mean", color="k")
        ax.fill_between(
            t_cont,
            realisations_percentiles[1],
            realisations_percentiles[3],
            alpha=0.5,
            label="68%",
        )
        ax.fill_between(
            t_cont,
            realisations_percentiles[0],
            realisations_percentiles[4],
            alpha=0.2,
            label="95%",
        )

        ax.errorbar(
            self.x,
            self.y,
            yerr=self.yerr,
            fmt="o",
            mfc="k",
            label="data",
        )
        ax.update(
            {
                "xlabel": "Time (days)",
                "ylabel": "Flux",
                "title": f"Posterior predictive checks - {n_samples} realisations",
            }
        )
        ax.legend(ncol=5, loc="best")
        fig.tight_layout()
        fig.savefig(
            f"{self.filename_prefix}_posterior_predictive_timeseries.png", dpi=300
        )

    def plot_timeseries_diagnostics_old(
        self, prediction_indexes: jax.Array | None = None, **kwargs
    ) -> None:
        """Plot the timeseries diagnostics.

        This function will call the :func:`plot_prediction` and :func:`plot_residuals` functions to
        plot the predicted timeseries and the residuals.

        Parameters
        ----------
        prediction_indexes : :obj:`jax.Array`, optional
            The prediction times, by default None
        **kwargs
            Additional keyword arguments to be passed to the :func:`plot_prediction` and :func:`plot_residuals` functions.
        """
        print("Plotting timeseries diagnostics...")
        log_transform = (
            False
            if (
                (not (self.process.use_tinygp or self.process.use_celerite))
                and self.process.log_transform
            )
            else None
        )

        (
            self.predictive_mean,
            self.predictive_cov,
        ) = self.process.compute_predictive_distribution(
            log_transform=log_transform, prediction_indexes=prediction_indexes
        )
        self.x_pred = (
            self.process.prediction_indexes.flatten()
            if prediction_indexes is None
            else prediction_indexes
        )
        fig, ax = plot_prediction(
            x=self.x.flatten(),
            y=self.y.flatten(),
            yerr=self.yerr.flatten(),
            x_pred=self.x_pred.flatten(),
            y_pred=self.predictive_mean.flatten(),
            cov_pred=self.predictive_cov,
            filename=self.filename_prefix,
            log_transform=self.process.log_transform,
            **kwargs,
        )

        (
            prediction_at_observation_times,
            _,
        ) = self.process.compute_predictive_distribution(
            log_transform=log_transform, prediction_indexes=self.x
        )

        fig2, ax2 = plot_residuals(
            x=self.x.flatten(),
            y=self.y.flatten(),
            yerr=self.yerr.flatten(),
            y_pred=prediction_at_observation_times.flatten(),
            filename=self.filename_prefix,
            log_transform=self.process.log_transform,
            **kwargs,
        )

    def posterior_predictive_checks(
        self,
        samples: jax.Array,
        plot_PSD: bool = True,
        plot_ACVF: bool = False,
        **kwargs,
    ):
        """Plot the posterior predictive checks.

        Parameters
        ----------
        samples : :obj:`jax.Array`
            The samples from the posterior distribution.
        plot_PSD : :obj:`bool`, optional
            Plot the posterior predictive PSDs, by default True
        plot_ACVF : :obj:`bool`, optional
            Plot the posterior predictive ACVFs, by default True
        **kwargs
            Additional keyword arguments.
            frequencies : jnp.ndarray, optional The frequencies at which to evaluate the PSDs of CARMA process, by default self.frequencies
            plot_lombscargle : bool, optional Plot the Lomb-Scargle periodogram, by default False
        """
        if isinstance(self.process, CARMAProcess):
            if self.process.p > 1:
                print("Converting CARMA samples to coefficients...")
                alpha = [
                    quad_to_coeff(samples[i, 1 : self.process.p + 1])
                    for i in range(samples.shape[0])
                ]
                sigma = samples[:, 0]
                roots = [
                    jnp.unique(jnp.roots(alpha[i]))[::-1]
                    for i in range(samples.shape[0])
                ]
                if self.process.q > 0:
                    if self.process.use_beta:
                        beta = samples[
                            :, self.process.p + 1 : self.process.p + 1 + self.process.q
                        ]
                    else:
                        beta = [
                            MA_quad_to_coeff(
                                self.process.q,
                                samples[
                                    i,
                                    self.process.p
                                    + 1 : self.process.p
                                    + 1
                                    + self.process.q,
                                ],
                            )
                            for i in range(samples.shape[0])
                        ]
            elif self.process.p == 1:
                alpha = samples[:, 1]
                sigma = samples[:, 0]
        else:
            if self.process.estimate_variance:
                index_variance = (self.process.model.parameters.free_names).index("var")
                variance = samples[:, index_variance]
            if self.process.scale_errors:
                index_nu = (self.process.model.parameters.free_names).index("nu")
                nu = samples[:, index_nu]
            if self.process.log_transform:
                index_const = (self.process.model.parameters.free_names).index("const")
                const = samples[:, index_const]
            params = samples

        posterior_ACVF = None
        # plot the posterior predictive PSDs
        if plot_PSD:
            print("Computing posterior predictive PSDs...")
            f = kwargs.get("frequencies", self.frequencies)
            if isinstance(self.process, CARMAProcess):
                if self.process.p == 1:
                    posterior_PSD = jnp.array(
                        [
                            sigma[i] / (alpha[i] ** 2 + 4 * jnp.pi**2 * f)
                            for i in range(samples.shape[0])
                        ]
                    )
                else:
                    if self.process.q > 0:
                        posterior_PSD = jnp.array(
                            [
                                CARMA_powerspectrum(f, alpha[i], beta[i], sigma[i])
                                for i in range(samples.shape[0])
                            ]
                        )
                    else:
                        posterior_PSD = jnp.array(
                            [
                                CARMA_powerspectrum(
                                    f,
                                    alpha[i],
                                    jnp.append(
                                        jnp.array([1]), jnp.zeros(self.process.p - 1)
                                    ),
                                    sigma[i],
                                )
                                for i in range(samples.shape[0])
                            ]
                        )
                print("Plotting posterior predictive PSDs...")
                plot_posterior_predictive_PSD(
                    f=f,
                    posterior_PSD=posterior_PSD,
                    x=self.x,
                    f_LS=self.frequencies,
                    y=self.y,
                    yerr=self.yerr,
                    filename=self.filename_prefix,
                    save_data=True,
                    **kwargs,
                )

            else:
                f_min = self.process.model.f0  # 0 is the first frequency
                f_max = self.process.model.fN
                f = jnp.logspace(jnp.log10(f_min), jnp.log10(f_max), 1000)

                posterior_PSD = []
                posterior_ACVF = []

                if isinstance(self.process.model, PSDToACV) and not (
                    self.process.use_tinygp or self.process.use_celerite
                ):  # when the PSD model is not approximated with tinygp or celerite
                    if self.process.estimate_variance:
                        sumP = np.array([])
                        if not os.path.isfile(
                            f"{self.filename_prefix}_normalisation_factor.txt"
                        ):
                            for it in range(samples.shape[0]):
                                self.process.model.parameters.set_free_values(
                                    samples[it]
                                )
                                R, factor = self.process.model.calculate(
                                    self.tau, with_ACVF_factor=True
                                )
                                sumP = np.append(sumP, factor)
                                # posterior_ACVF.append(R)
                                P = (
                                    self.process.model.PSD.calculate(f)
                                    / factor
                                    * variance[it]
                                )

                                posterior_PSD.append(P)
                                print(f"Sample: {it+1}/{samples.shape[0]}", end="\r")
                                sys.stdout.flush()
                            np.savetxt(
                                f"{self.filename_prefix}_normalisation_factor.txt", sumP
                            )

                        else:
                            print(
                                "Normalisation factor already computed, loading it..."
                            )
                            factors = np.loadtxt(
                                f"{self.filename_prefix}_normalisation_factor.txt"
                            )

                            for it in range(samples.shape[0]):
                                self.process.model.parameters.set_free_values(
                                    samples[it]
                                )
                                P = (
                                    self.process.model.PSD.calculate(f)
                                    / factors[it]
                                    * variance[it]
                                )  # self.process.model.frequencies[1:])
                                posterior_PSD.append(P)

                                print(f"Sample: {it+1}/{samples.shape[0]}", end="\r")
                                sys.stdout.flush()

                    else:
                        for it in range(samples.shape[0]):
                            self.process.model.parameters.set_free_values(samples[it])
                            posterior_PSD.append(self.process.model.PSD.calculate(f))
                            print(f"Samples: {it+1}/{samples.shape[0]}", end="\r")
                            sys.stdout.flush()

                    posterior_PSD = np.array(posterior_PSD)
                    f_LS = self.frequencies

                    plot_posterior_predictive_PSD(
                        f=f,
                        posterior_PSD=posterior_PSD,
                        x=self.x,
                        y=np.log(self.y - np.mean(const))
                        if self.process.log_transform
                        else self.y,
                        yerr=self.yerr,
                        filename=self.filename_prefix,
                        save_data=True,
                        f_LS=f_LS,
                        f_min_obs=self.f_min,
                        f_max_obs=self.f_max,
                        **kwargs,
                    )

                # when the PSD model is approximated with tinygp, it is normalised to the variance
                elif isinstance(self.process.model, PSDToACV) and (
                    self.process.use_tinygp or self.process.use_celerite
                ):
                    posterior_PSD_approx = []
                    if self.process.model.method == "SHO":
                        Power_spectrum_model = SHO_power_spectrum
                        norm = lambda a, f: jnp.sum(a * f)
                    elif self.process.model.method == "DRWCelerite":
                        Power_spectrum_model = DRWCelerite_power_spectrum
                        norm = lambda a, f: jnp.sum(a * f * 2 * jnp.pi / 3)

                    print("Plotting posterior predictive PSDs with tinygp...")
                    fc = self.process.model.spectral_points
                    factors = []
                    for it in range(samples.shape[0]):
                        self.process.model.parameters.set_free_values(samples[it])
                        amps, _ = self.process.model.get_approx_coefs()
                        factors.append(norm(amps, fc))

                        psd = self.process.model.PSD.calculate(f)  # calculate the PSD
                        psd_samples = Power_spectrum_model(
                            f, amps[..., None], fc[..., None]
                        ).sum(axis=0)

                        psd /= psd[0]  # normalise the PSD to the first value
                        psd_samples /= psd_samples[
                            0
                        ]  # normalise the PSD to the first value

                        psd /= norm(amps, fc)
                        # normalise the PSD to the sum of the amplitudes
                        psd_samples /= norm(amps, fc)
                        # normalise the PSD to the sum of the amplitudes

                        psd *= variance[it]  # scale the PSD to the variance
                        psd_samples *= variance[it]  # scale the PSD to the variance

                        posterior_PSD.append(psd)
                        posterior_PSD_approx.append(psd_samples)
                        print(f"Samples: {it+1}/{samples.shape[0]}", end="\r")
                        sys.stdout.flush()

                    factors = np.array(factors)
                    np.savetxt(
                        f"{self.filename_prefix}_normalisation_factor.txt", factors
                    )
                    posterior_PSD = np.array(posterior_PSD)
                    posterior_PSD_approx = np.array(posterior_PSD_approx)
                    f_LS = self.frequencies

                    plot_posterior_predictive_PSD(
                        f=f,
                        posterior_PSD=posterior_PSD,
                        x=self.x,
                        y=np.log(self.y) if self.process.log_transform else self.y,
                        yerr=self.yerr,
                        filename=self.filename_prefix,
                        save_data=True,
                        posterior_PSD_approx=posterior_PSD_approx,
                        f_LS=f_LS,
                        f_min_obs=self.f_min,
                        f_max_obs=self.f_max,
                        **kwargs,
                    )

            # plot the posterior predictive PSDs

        # plot the posterior predictive ACFs
        if plot_ACVF:
            print("Computing posterior predictive ACFs...")
            if isinstance(self.process, CARMAProcess):
                if self.process.p == 1:
                    posterior_ACVF = jnp.array(
                        [
                            0.5 * sigma[i] / alpha[i] * jnp.exp(-alpha[i] * self.tau)
                            for i in range(samples.shape[0])
                        ]
                    )
                else:
                    if self.process.q > 0:
                        posterior_ACVF = jnp.array(
                            [
                                CARMA_autocovariance(
                                    self.tau, roots[i], beta[i], sigma[i]
                                )
                                for i in range(samples.shape[0])
                            ]
                        )
                    else:
                        posterior_ACVF = jnp.array(
                            [
                                CARMA_autocovariance(
                                    self.tau, roots[i], jnp.array([1]), sigma[i]
                                )
                                for i in range(samples.shape[0])
                            ]
                        )
            elif isinstance(self.process.model, PSDToACV):
                raise NotImplementedError(
                    "Posterior predictive ACFs are not implemented for PSD models."
                )
                # pass
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
                raise NotImplementedError(
                    "Posterior predictive ACFs are not implemented for Gaussian processes."
                )

            posterior_ACVF /= posterior_ACVF[:, 0, None]

            print("Plotting posterior predictive ACFs...")
            plot_posterior_predictive_ACF(
                tau=self.tau,
                acf=posterior_ACVF,
                x=self.x,
                y=self.y,
                filename=self.filename_prefix,
                save_data=True,
                **kwargs,
            )
