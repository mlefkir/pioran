"""Infer the (hyper)parameters of processes."""
import json
import os
from functools import partial

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
from equinox import filter_jit

from .acvf import CovarianceFunction
from .carma.carma_core import CARMAProcess
from .core import GaussianProcess
from .plots import (diagnostics_psd_approx, plot_prior_predictive_PSD,
                    plot_priors_samples, residuals_quantiles,
                    violin_plots_psd_approx)
from .psdtoacv import PSDToACV
from .utils import (get_samples_psd, progress_bar_factory,
                    save_sampling_results, scalable_methods,
                    wrapper_psd_true_samples)
from .utils.mcmc_visualisations import (from_samples_to_inference_data,
                                        plot_diagnostics_sampling)

inference_methods = ["ultranest", "blackjax_nuts","blackjax_DYHMC"]
"""List of inference methods implemented."""

_USE_BLACKJAX = True
_USE_ULTRANEST = True
_USE_MPI = True

# check if the optional inference packages are installed
try:
    import asdf
    import optax
    import blackjax  # for the HMC/MCMC sampling
    import tqdm  # for the progress bar
    from blackjax.diagnostics import (effective_sample_size,
                                      potential_scale_reduction)
except ImportError:
    asdf = None
    blackjax = None
    tqdm = None
    _USE_BLACKJAX = False

try:
    import ultranest  # for the nested sampling
    import ultranest.stepsampler
except ImportError:
    ultranest = None
    _USE_ULTRANEST = False

# check if MPI is installed
try:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
except ImportError:
    rank = 0
    _USE_MPI = False


class Inference:
    r"""Class to infer the value of the (hyper)parameters of the Gaussian Process.

    Various methods to sample the posterior probability distribution of the (hyper)parameters of the Gaussian Process are implemented
    as wrappers around the inference packages `blackjax` and `ultranest`.


    Parameters
    ----------
    Process : :class:`~pioran.core.GaussianProcess`
        Process object.
    priors : :obj:`function`
        Function to define the priors for the (hyper)parameters.
    method : :obj:`str`, optional
        "NS": using nested sampling via ultranest
    n_samples_checks : :obj:`int`, optional
        Number of samples to take from the prior distribution, by default 1000
    seed_check : :obj:`int`, optional
        Seed for the random number generator, by default 0
    run_checks : :obj:`bool`, optional
        Run the prior predictive checks, by default True
    log_dir : :obj:`str`, optional
        Directory to save the results of the inference, by default 'log_dir'
    title_plots : :obj:`bool`, optional
        Plot the title of the figures, by default True

    Raises
    ------
    ImportError
        If the required packages are not installed.
    ValueError
        If the saved config file is different from the current config, or if the method is not valid.
    TypeError
        If the method is not a string.
    """
    process: GaussianProcess | CARMAProcess
    """Process object."""
    n_pars: int
    """Number of (hyper)parameters."""
    priors: callable
    """Function to define the priors for the (hyper)parameters."""
    log_dir: str
    """Directory to save the results of the inference."""
    plot_dir: str
    """Directory to save the plots of the inference."""
    method: str
    """Method to use for the inference."""

    def __init__(
        self,
        Process: GaussianProcess | CARMAProcess,
        priors,
        method,
        n_samples_checks=1000,
        seed_check=0,
        run_checks=True,
        log_dir="log_dir",
        title_plots=True
    ):
        r"""Constructor method for the Inference class."""

        self.process = Process
        self.n_pars = len(self.process.model.parameters.free_names)
        self.priors = priors
        self.log_dir = log_dir
        self.plot_dir = f"{self.log_dir}/plots"

        # create the log directory and save the config file
        if rank == 0:
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            if not os.path.exists(self.plot_dir):
                os.makedirs(self.plot_dir)
            if os.path.isfile(self.log_dir + "/config.json"):
                print(
                    ">>>>>> The config file already exists. Loading the previous config."
                )
                file = open(self.log_dir + "/config.json")
                dict_config_old = json.load(file)
                print(">>>>>> Check the config file:")
                dict_config_new = self.save_config(save_file=False)
                if dict_config_old == dict_config_new:
                    print(">>>>>> The config file is the identical.")
                else:
                    raise ValueError("The config file is different.")
            else:
                self.save_config()

        if isinstance(method, str):
            if method not in inference_methods:
                raise ValueError(
                    f"method must be one of {inference_methods}, got {method}"
                )
            self.method = method

            # check if the required packages are installed
            if (method == "blackjax_nuts" or method == "blackjax_DYHMC") and not _USE_BLACKJAX:
                raise ImportError(
                    "blackjax and/or asdf not installed. Please install them to use blackjax."
                )

            elif method == "ultranest" and not _USE_ULTRANEST:
                raise ImportError(
                    "ultranest not installed. Please install it to use nested sampling."
                )
        else:
            raise TypeError("method must be a string.")

        # run prior predictive checks
        if run_checks and rank == 0:
            self.prior_predictive_checks(n_samples_checks, seed_check)

            if isinstance(Process, GaussianProcess) and isinstance(
                self.process.model, PSDToACV
            ):
                if self.process.model.method in scalable_methods:
                    print(
                        f"\n>>>>>> The PSD model is a {self.process.model.method} decomposition, checking the approximation."
                    )
                    self.check_approximation(n_samples_checks, seed_check,title=title_plots)

        if _USE_MPI:
            comm.Barrier()

    def save_config(self, save_file=True):
        """Save the configuration of the inference.

        Save the configuration of the inference, process and model in a json file.

        Parameters
        ----------
        save_file : :obj:`bool`, optional

        Returns
        -------
        dict_config : :obj:`dict`
            Dictionary with the configuration of the inference, process and model.
        """
        dict_config = {}

        # type of process
        if isinstance(self.process, GaussianProcess):
            dict_config["process"] = {"type": "GaussianProcess", "options": {}}
            dict_config["process"]["options"][
                "estimate_mean"
            ] = self.process.estimate_mean
        elif isinstance(self.process, CARMAProcess):
            dict_config["process"] = {"type": "CARMAProcess", "options": {}}
            dict_config["process"]["options"]["use_beta"] = self.process.use_beta
        else:
            dict_config["process"] = {"type": "unknown", "options": {}}

        # options of the process
        dict_config["process"]["options"]["estimate_mean"] = self.process.estimate_mean
        dict_config["process"]["options"]["scale_errors"] = self.process.scale_errors
        dict_config["process"]["options"][
            "estimate_variance"
        ] = self.process.estimate_variance
        dict_config["process"]["options"]["log_transform"] = self.process.log_transform

        # type of model
        if isinstance(self.process.model, PSDToACV):
            dict_config["model"] = {"type": "PowerSpectralDensity", "info": {}}
            dict_config["model"]["info"][
                "expression"
            ] = self.process.model.PSD.expression
            dict_config["model"]["info"]["method"] = self.process.model.method
            dict_config["model"]["info"]["S_low"] = self.process.model.S_low
            dict_config["model"]["info"]["S_high"] = self.process.model.S_high
            if not ("FFT" in self.process.model.method):
                dict_config["model"]["info"][
                    "n_components"
                ] = self.process.model.n_components

        elif isinstance(self.process.model, CovarianceFunction):
            dict_config["model"] = {"type": "CovarianceFunction", "info": {}}
            dict_config["model"]["info"]["expression"] = self.process.model.expression
        elif isinstance(self.process, CARMAProcess):
            dict_config["model"] = {"type": "CARMA_model", "info": {}}
            dict_config["model"]["info"]["p"] = self.process.p
            dict_config["model"]["info"]["q"] = self.process.q

        # parameters of the model
        dict_config["model"]["parameters"] = {}
        dict_config["model"]["parameters"][
            "names"
        ] = self.process.model.parameters.names
        dict_config["model"]["parameters"][
            "free_parameters"
        ] = self.process.model.parameters.free_names

        # data_related
        dict_config["data"] = {}
        dict_config["data"]["nb_observation_points"] = len(
            self.process.observation_indexes.flatten()
        )
        dict_config["data"]["duration"] = float(
            self.process.observation_indexes.flatten()[-1] - self.process.observation_indexes.flatten()[0]
        )
        dict_config["data"]["min_sampling"] = np.diff(
            self.process.observation_indexes.flatten()
        ).min()
        dict_config["data"][
            "observation_indexes"
        ] = self.process.observation_indexes.flatten().tolist()
        dict_config["data"][
            "observation_values"
        ] = self.process.observation_values.flatten().tolist()
        dict_config["data"][
            "observation_errors"
        ] = self.process.observation_errors.flatten().tolist()

        if save_file:
            with open(self.log_dir + "/config.json", "w", encoding="utf-8") as f:
                json.dump(dict_config, f, ensure_ascii=False, indent=4)

        return dict_config

    def prior_predictive_checks(
        self,
        n_samples_checks,
        seed_check,
        n_frequencies=1000,
        plot_prior_samples=True,
        plot_prior_predictive_distribution=True,
    ):
        """Check the prior predictive distribution.

        Get samples from the prior distribution and plot them, and calculate the prior predictive
        distribution of the model and plot it.

        Parameters
        ----------
        n_samples_checks : :obj:`int`
            Number of samples to take from the prior distribution, by default 1000
        seed_check : :obj:`int`
            Seed for the random number generator
        plot_prior_samples : :obj:`bool`, optional
            Plot the prior samples, by default True
        plot_prior_predictive_distributions : :obj:`bool`, optional
            Plot the prior predictive distribution of the model, by default True

        """
        key = jax.random.PRNGKey(seed_check)
        freqs = jnp.geomspace(
            self.process.model.f0, self.process.model.fN, n_frequencies
        )

        if self.method == "ultranest":
            # draw samples from the prior distribution
            uniform_samples = jax.random.uniform(
                key=key, shape=(self.n_pars, n_samples_checks)
            )
            self.params_samples = self.priors(np.array(uniform_samples))  # [indexes]
        else:
            raise NotImplementedError("Only ultranest is implemented for now.")

        if plot_prior_samples:
            fig, _ = plot_priors_samples(
                self.params_samples, self.process.model.parameters.free_names
            )
            fig.savefig(f"{self.plot_dir}/prior_samples.pdf")

        if plot_prior_predictive_distribution:
            # if model is PSD plot predictive PSD and ACVF
            if isinstance(self.process.model, PSDToACV):
                psd_true_samples = wrapper_psd_true_samples(
                    self.process.model, freqs, self.params_samples.T
                )
                #
                if not self.process.estimate_variance:
                    fig, _ = plot_prior_predictive_PSD(freqs, psd_true_samples)
                # else:
                # psd_samples = psd_true_samples/psd_true_samples[...,0,None]
                # idx_var = self.process.model.parameters.names.index('var')
                # fig,_ = plot_prior_predictive_PSD(freqs,jnp.transpose(psd_samples.T*self.params_samples[idx_var].flatten()))
                # fig.savefig("prior_predictive_psd_distribution.pdf")
            # psd_samples = wrapper_psd_true_samples(self.process.model,freqs,self.params_samples)
            #     fig,_ = plot_prior_predictive_distribution(params_samples)
            #     fig.savefig("prior_predictive_distribution.pdf")

    def check_approximation(
        self,
        n_samples_checks: int,
        seed_check: int,
        n_frequencies: int = 1000,
        plot_diagnostics: bool = True,
        plot_violins: bool = True,
        plot_quantiles: bool = True,
        title:bool= True
    ):
        """Check the approximation of the PSD with the kernel decomposition.

        This method will take random samples from the prior distribution and compare the PSD obtained
        with the SHO decomposition with the true PSD.

        Parameters
        ----------
        n_samples_checks : :obj:`int`
            Number of samples to take from the prior distribution, by default 1000
        seed_check : :obj:`int`
            Seed for the random number generator
        n_frequencies : :obj:`int`, optional
            Number of frequencies to evaluate the PSD, by default 1000
        plot_diagnostics : :obj:`bool`, optional
            Plot the diagnostics of the approximation, by default True
        plot_violins : :obj:`bool`, optional
            Plot the violin plots of the residuals and the ratios, by default True
        plot_quantiles : :obj:`bool`, optional
            Plot the quantiles of the residuals and the ratios, by default True
        plot_prior_samples : :obj:`bool`, optional
            Plot the prior samples, by default True
        title : :obj:`bool`, optional
            Plot the title of the figure, by default True

        Returns
        -------
        figs : :obj:`list`
            List of figures.
        residuals : :obj:`jax.Array`
            Residuals of the PSD approximation.
        ratio : :obj:`jax.Array`
            Ratio of the PSD approximation.
        """
        freqs = jnp.geomspace(
            self.process.model.f0, self.process.model.fN, n_frequencies
        )

        # get the true PSD and the SHO PSD samples
        psd_true, psd_approx = get_samples_psd(
            self.process.model, freqs, self.params_samples.T
        )
        # compute the residuals and the ratios
        residuals = psd_true - psd_approx
        ratio = psd_approx / psd_true
        figs = []

        if plot_diagnostics:
            fig, _ = diagnostics_psd_approx(
                f=freqs,
                res=residuals,
                ratio=ratio,
                f_min=self.process.model.f_min_obs,
                f_max=self.process.model.f_max_obs
            )
            fig.savefig(f"{self.plot_dir}/diagnostics_psd_approx.pdf")
            figs.append(fig)

        if plot_violins:
            fig, _ = violin_plots_psd_approx(res=residuals, ratio=ratio,title=title)
            fig.savefig(f"{self.plot_dir}/violin_plots_psd_approx.pdf")
            figs.append(fig)

        if plot_quantiles:
            fig, _ = residuals_quantiles(
                residuals=residuals,
                ratio=ratio,
                f=freqs,
                f_min=self.process.model.f_min_obs,
                f_max=self.process.model.f_max_obs,
                title=title
            )
            fig.savefig(f"{self.plot_dir}/quantiles_psd_approx.pdf")
            figs.append(fig)

        return figs, residuals, ratio

    def run(
        self,
        verbose: bool = True,
        user_log_likelihood=None,
        seed: int = 0,
        n_chains: int = 1,
        n_samples: int = 1_000,
        n_warmup_steps: int = 1_000,
        use_stepsampler: bool = False,
    ):
        """Estimate the (hyper)parameters of the Gaussian Process.

        Run the inference method.

        Parameters
        ----------
        verbose : :obj:`bool`, optional
            Be verbose, by default True
        user_log_likelihood : :obj:`function`, optional
            User-defined function to compute the log-likelihood, by default None
        seed : :obj:`int`, optional
            Seed for the random number generator, by default 0
        n_chains : :obj:`int`, optional
            Number of chains, by default 1
        n_samples : :obj:`int`, optional
            Number of samples to take from the posterior distribution, by default 1_000
        n_warmup_steps : :obj:`int`, optional
            Number of warmup steps, by default 1_000
        use_stepsampler : :obj:`bool`, optional
            Use the slice sampler as step sampler, by default False

        Returns
        -------
        results: dict
            Results of the sampling. The keys differ depending on the method/sampler used.
        """
        rng_key = jax.random.PRNGKey(seed)

        log_likelihood = (
            filter_jit(self.process.wrapper_log_marginal_likelihood)
            if user_log_likelihood is None
            else user_log_likelihood
        )

        if self.method == "ultranest":
            results, sampler = self.nested_sampling(
                priors=self.priors,
                log_likelihood=log_likelihood,
                verbose=verbose,
                use_stepsampler=use_stepsampler,
            )

            # make sure all the processes are done
            if _USE_MPI:
                comm.Barrier()
            self.process.model.parameters.set_free_values(
                results["maximum_likelihood"]["point"]
            )  # results['posterior']['median'])
            if rank == 0:
                print("\n>>>>>> Plotting corner and trace.")
                sampler.plot()

        elif self.method == "blackjax_nuts":
            if os.path.isfile(f"{self.log_dir}/chains.asdf"):
                print(
                    "The sampling results file already exists. Loading the previous results."
                )
                af = asdf.open(f"{self.log_dir}/chains.asdf")
                # check that warmup and sampling parameters are the same
                loaded_info = af["info"]
                assert (
                    loaded_info["sampler"] == "NUTS"
                ), f"The sampler saved ({loaded_info['sampler']}) is different from the one given ('NUTS')."
                assert (
                    loaded_info["package"] == "blackjax"
                ), f"The package saved ({loaded_info['package']}) is different from the one given ('blackjax')."
                assert (
                    loaded_info["num_warmup"] == n_warmup_steps
                ), f"The number of warmup steps saved ({loaded_info['num_warmup']}) is different from the one given ({n_warmup_steps})."
                assert (
                    loaded_info["num_samples"] == n_samples
                ), f"The number of samples saved ({loaded_info['num_samples']}) is different from the one given ({n_samples})."
                assert (
                    loaded_info["num_chains"] == n_chains
                ), f"The number of chains saved ({loaded_info['num_chains']}) is different from the one given ({n_chains})."

                samples = np.array(
                    [
                        af["samples"][f"chain_{i}"]
                        for i in range(af["info"]["num_chains"])
                    ]
                )
                log_prob = np.array(
                    [
                        af["log_prob"][f"chain_{i}"]
                        for i in range(af["info"]["num_chains"])
                    ]
                )
            else:
                print("\n>>>>>> Sampling the posterior distribution.")
                samples, log_prob = self.blackjax_NUTS(
                    rng_key=rng_key,
                    initial_position=self.process.model.parameters.free_values,
                    log_likelihood=log_likelihood,
                    log_prior=self.priors,
                    num_warmup_steps=n_warmup_steps,
                    num_samples=n_samples,
                    num_chains=n_chains,
                )

            names = self.process.model.parameters.free_names
            dataset = from_samples_to_inference_data(names, samples)
            print("\n>>>>>> Summary of the sampling.")
            print(az.summary(dataset))
            print("\n>>>>>> Plotting corner, trace and diagnostics.")
            plot_diagnostics_sampling(
                dataset, plot_dir=self.plot_dir, prefix="blackjax_nuts_"
            )

            medians = jnp.median(samples, axis=(0, 2))
            self.process.model.parameters.set_free_values(medians)
            results = {
                "samples": samples,
                "log_prob": log_prob,
                "inferencedata": dataset,
            }
        elif self.method == "blackjax_DYHMC":
            if os.path.isfile(f"{self.log_dir}/chains.asdf"):
                print(
                    "The sampling results file already exists. Loading the previous results."
                )
                af = asdf.open(f"{self.log_dir}/chains.asdf")
                # check that warmup and sampling parameters are the same
                loaded_info = af["info"]
                assert (
                    loaded_info["sampler"] == "DYHMC"
                ), f"The sampler saved ({loaded_info['sampler']}) is different from the one given ('NUTS')."
                assert (
                    loaded_info["package"] == "blackjax"
                ), f"The package saved ({loaded_info['package']}) is different from the one given ('blackjax')."
                assert (
                    loaded_info["num_warmup"] == n_warmup_steps
                ), f"The number of warmup steps saved ({loaded_info['num_warmup']}) is different from the one given ({n_warmup_steps})."
                assert (
                    loaded_info["num_samples"] == n_samples
                ), f"The number of samples saved ({loaded_info['num_samples']}) is different from the one given ({n_samples})."
                assert (
                    loaded_info["num_chains"] == n_chains
                ), f"The number of chains saved ({loaded_info['num_chains']}) is different from the one given ({n_chains})."

                samples = np.array(
                    [
                        af["samples"][f"chain_{i}"]
                        for i in range(af["info"]["num_chains"])
                    ]
                )
                log_prob = np.array(
                    [
                        af["log_prob"][f"chain_{i}"]
                        for i in range(af["info"]["num_chains"])
                    ]
                )
            else:
                print("\n>>>>>> Sampling the posterior distribution.")
                samples, log_prob = self.blackjax_DYHMC(
                    rng_key=rng_key,
                    initial_position=self.process.model.parameters.free_values,
                    log_likelihood=log_likelihood,
                    log_prior=self.priors,
                    num_warmup_steps=n_warmup_steps,
                    num_samples=n_samples,
                    num_chains=n_chains
                )
        else:
            raise NotImplementedError(f"Only {inference_methods} is implemented for now.")

        print(self.process)
        return results
    def blackjax_DYHMC(
        self,
        rng_key: jax.random.PRNGKey,
        initial_position: jax.Array,
        log_likelihood: callable,
        log_prior: callable,
        num_warmup_steps: int = 1_000,
        num_samples: int = 1_000,
        num_chains: int = 1,
        step_size: float = 1e-2,
        learning_rate: float = 1e-2,
    ):
        """Sample the posterior distribution using the NUTS sampler from blackjax.

        Wrapper around the NUTS sampler from blackjax to sample the posterior distribution.
        This function also performs the warmup via window adaptation.

        Parameters
        ----------
        rng_key : :obj:`jax.random.PRNGKey`
            Random key for the random number generator.
        initial_position : :obj:`jax.Array`
            Initial position of the chains.
        log_likelihood : :obj:`function`
            Function to compute the log-likelihood.
        log_prior : :obj:`function`
            Function to compute the log-prior.
        num_warmup_steps : :obj:`int`, optional
            Number of warmup steps, by default 1_000
        num_samples : :obj:`int`, optional
            Number of samples to take from the posterior distribution, by default 1_000
        num_chains : :obj:`int`, optional
            Number of chains, by default 1

        Returns
        -------
        samples : :obj:`jax.Array`
            Samples from the posterior distribution. It has shape (num_chains, num_params, num_samples).
        log_prob : :obj:`jax.Array`
            Log-probability of the samples.
        """

        log_posterior = jax.jit(lambda x: log_likelihood(x) + log_prior(x))

        # set the initial positions of the chains
        initial_positions = jnp.array(initial_position) * jnp.ones(
            (num_chains, self.n_pars)
        )
        # warmup loop
        keys_adapt, inference_key = jax.random.split(rng_key, 2)


        warmup = blackjax.chees_adaptation(log_posterior, num_chains=num_chains, target_acceptance_rate=0.75)
    
        (state, parameters), warmup_info = warmup.run(
            keys_adapt,
            initial_positions,
            step_size=step_size,
            optim=optax.adamw(learning_rate=learning_rate),
            num_steps=num_warmup_steps
        )

        steps_size = parameters["step_size"].block_until_ready()
        parameters["inverse_mass_matrix"] = np.array(parameters["inverse_mass_matrix"])
        parameters["step_size"] = np.array(parameters["step_size"])

        # save the warmup state
        warmup = {
            "parameters": parameters,
            "state.position": np.array(state.position),
            "state.logdensity": np.array(state.logdensity),
            "state.logdensity_grad": np.array(state.logdensity_grad),
        }

        print(f"\n>>>>>> Warmup done. Steps size: {steps_size}")

        print(f"\n>>>>>> Sampling the posterior distribution.{warmup}")
        print(f"\n>>>>>> Sampling the posterior distribution.{warmup_info}")
        print(f"\n>>>>>> Sampling the posterior distribution.{state}")
        
        # inference loop
        # @partial(jax.jit, static_argnums=(3, 4))
        # def inference_loop(rng_key, parameters, initial_state, num_samples, num_chains):
        #     kernel = blackjax.dynamic_hmc(log_posterior, **parameters).step

        #     @jax.jit
        #     # @progress_bar_factory(num_samples, num_chains)
        #     def one_step(carry, iter_num):
        #         state, rng_key = carry
        #         key = rng_key[iter_num]

        #         state, _ = kernel(key, state)
        #         return (state, rng_key), state

        #     keys = jax.random.split(rng_key, num_samples)

        #     carry = (initial_state, keys)
        #     _, states = jax.lax.scan(one_step, carry, jnp.arange(num_samples))
        #     return states

        # # split the random keys for the chains
        # keys = jax.random.split(rng_key, num_chains)
        # # pmap the inference loop
        # inference_loop_multiple_chains = jax.pmap(
        #     inference_loop,
        #     in_axes=(0, 0, 0, None, None),
        #     static_broadcasted_argnums=(3, 4),
        # )
        # run the inference loop
        # states = inference_loop_multiple_chains(
        #     keys, parameters, state, num_samples, num_chains
        # )
        # get the samples and the log-probability
        
        chain_keys = jax.random.split(inference_key, num_chains)
        kernel = blackjax.dynamic_hmc(log_posterior, **parameters)

        states_f, states, infos = jax.vmap(
        lambda key, state: blackjax.util.run_inference_algorithm(key, state, kernel, num_samples)
        )(chain_keys, state)

        samples = states.position.block_until_ready()
        log_prob = states.logdensity.block_until_ready()
        log_density_grad = states.logdensity_grad.block_until_ready()
        print(f"\n>>>>>> Sampling the posterior distribution.{samples}")
        print(f"\n>>>>>> Sampling the posterior distribution.{log_prob}")
        print(f"\n>>>>>> Sampling the posterior distribution.{log_density_grad}")
        # save the sampling results
        info = {
            "sampler": "DYHMC",
            "package": "blackjax",
            "package_version": blackjax.__version__,
            "n_params": self.n_pars,
            "num_samples": num_samples,
            "num_warmup": num_warmup_steps,
            "num_chains": num_chains,
            "ESS": np.array(effective_sample_size(samples.T)),
            "Rhat-split": np.array(potential_scale_reduction(samples.T)),
        }
        print(f"\n>>>>>> Sampling the posterior distribution.{info}")
        samples = jnp.transpose(samples, axes=(0, 2, 1))

        save_sampling_results(
            info=info,
            warmup=warmup,
            samples=samples,
            log_prob=log_prob,
            log_densitygrad=log_density_grad,
            filename=f"{self.log_dir}/chains",
        )

        return samples, log_prob

    def blackjax_NUTS(
        self,
        rng_key: jax.random.PRNGKey,
        initial_position: jax.Array,
        log_likelihood: callable,
        log_prior: callable,
        num_warmup_steps: int = 1_000,
        num_samples: int = 1_000,
        num_chains: int = 1,
    ):
        """Sample the posterior distribution using the NUTS sampler from blackjax.

        Wrapper around the NUTS sampler from blackjax to sample the posterior distribution.
        This function also performs the warmup via window adaptation.

        Parameters
        ----------
        rng_key : :obj:`jax.random.PRNGKey`
            Random key for the random number generator.
        initial_position : :obj:`jax.Array`
            Initial position of the chains.
        log_likelihood : :obj:`function`
            Function to compute the log-likelihood.
        log_prior : :obj:`function`
            Function to compute the log-prior.
        num_warmup_steps : :obj:`int`, optional
            Number of warmup steps, by default 1_000
        num_samples : :obj:`int`, optional
            Number of samples to take from the posterior distribution, by default 1_000
        num_chains : :obj:`int`, optional
            Number of chains, by default 1

        Returns
        -------
        samples : :obj:`jax.Array`
            Samples from the posterior distribution. It has shape (num_chains, num_params, num_samples).
        log_prob : :obj:`jax.Array`
            Log-probability of the samples.
        """

        log_posterior = jax.jit(lambda x: log_likelihood(x) + log_prior(x))

        # set the initial positions of the chains
        initial_positions = jnp.array(initial_position) * jnp.ones(
            (num_chains, self.n_pars)
        )
        # warmup loop
        keys_adapt = jax.random.split(rng_key, num_chains)

        warmup = blackjax.window_adaptation(
            blackjax.nuts, log_posterior, progress_bar=True)#, num_chains=num_chains
    
        warmup_run = partial(warmup.run, num_steps=num_warmup_steps)
        (state, parameters), _ = jax.pmap(warmup_run, in_axes=(0, 0))(
            keys_adapt, initial_positions
        )
        steps_size = parameters["step_size"].block_until_ready()

        parameters["inverse_mass_matrix"] = np.array(parameters["inverse_mass_matrix"])
        parameters["step_size"] = np.array(parameters["step_size"])

        # save the warmup state
        warmup = {
            "parameters": parameters,
            "state.position": np.array(state.position),
            "state.logdensity": np.array(state.logdensity),
            "state.logdensity_grad": np.array(state.logdensity_grad),
        }

        print(f"\n>>>>>> Warmup done. Steps size: {steps_size}")

        # inference loop
        @partial(jax.jit, static_argnums=(3, 4))
        def inference_loop(rng_key, parameters, initial_state, num_samples, num_chains):
            kernel = blackjax.nuts(log_posterior, **parameters).step

            @jax.jit
            @progress_bar_factory(num_samples, num_chains)
            def one_step(carry, iter_num):
                state, rng_key = carry
                key = rng_key[iter_num]

                state, _ = kernel(key, state)
                return (state, rng_key), state

            keys = jax.random.split(rng_key, num_samples)

            carry = (initial_state, keys)
            _, states = jax.lax.scan(one_step, carry, jnp.arange(num_samples))
            return states

        # split the random keys for the chains
        keys = jax.random.split(rng_key, num_chains)
        # pmap the inference loop
        inference_loop_multiple_chains = jax.pmap(
            inference_loop,
            in_axes=(0, 0, 0, None, None),
            static_broadcasted_argnums=(3, 4),
        )
        # run the inference loop
        states = inference_loop_multiple_chains(
            keys, parameters, state, num_samples, num_chains
        )
        # get the samples and the log-probability
        samples = states.position.block_until_ready()
        log_prob = states.logdensity.block_until_ready()
        log_density_grad = states.logdensity_grad.block_until_ready()

        # save the sampling results
        info = {
            "sampler": "NUTS",
            "package": "blackjax",
            "package_version": blackjax.__version__,
            "n_params": self.n_pars,
            "num_samples": num_samples,
            "num_warmup": num_warmup_steps,
            "num_chains": num_chains,
            "ESS": np.array(effective_sample_size(samples.T)),
            "Rhat-split": np.array(potential_scale_reduction(samples.T)),
        }
        samples = jnp.transpose(samples, axes=(0, 2, 1))

        save_sampling_results(
            info=info,
            warmup=warmup,
            samples=samples,
            log_prob=log_prob,
            log_densitygrad=log_density_grad,
            filename=f"{self.log_dir}/chains",
        )

        return samples, log_prob

    def nested_sampling(
        self,
        priors: callable,
        log_likelihood: callable,
        verbose: bool = True,
        use_stepsampler: bool = False,
        resume: bool = True,
        run_kwargs={},
        slice_steps=100,
    ):
        r"""Sample the posterior distribution of the (hyper)parameters of the Gaussian Process with nested sampling via ultranest.

        Perform nested sampling to sample the (hyper)parameters of the Gaussian Process.

        Parameters
        ----------
        priors : :obj:`function`
            Function to define the priors for the parameters
        log_likelihood : :obj:`function`
            Function to compute the log-likelihood.
        verbose : :obj:`bool`, optional
            Print the results of the sample and the progress of the sampling, by default True
        use_stepsampler : :obj:`bool`, optional
            Use the slice sampler as step sampler, by default False
        resume : :obj:`bool`, optional
            Resume the sampling from the previous run, by default True
        run_kwargs : :obj:`dict`, optional
            Dictionary of arguments for ReactiveNestedSampler.run() see https://johannesbuchner.github.io/UltraNest/ultranest.html#module-ultranest.integrator
        slice_steps : :obj:`int`, optional
            Number of steps for the slice sampler, by default 100

        Returns
        -------
        results: dict
            Dictionary of results from the nested sampling.
        """

        viz = {} if verbose else {"show_status": False, "viz_callback": void}
        free_names = self.process.model.parameters.free_names
        sampler = ultranest.ReactiveNestedSampler(
            free_names, log_likelihood, priors, resume=resume, log_dir=self.log_dir
        )
        if use_stepsampler:
            sampler.stepsampler = ultranest.stepsampler.SliceSampler(
                nsteps=slice_steps,
                generate_direction=ultranest.stepsampler.generate_mixture_random_direction,
            )

        if verbose:
            results = sampler.run(**viz)
        else:
            results = sampler.run(**run_kwargs, **viz)
        return results, sampler


def void(*args, **kwargs):
    """Void function to avoid printing the status of the nested sampling."""
    pass
