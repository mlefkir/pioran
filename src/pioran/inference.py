"""Class and functions for inference with Gaussian Processes and other methods.

"""
from typing import Union

import os
import jax
import json
import jax.numpy as jnp
import numpy as np
import ultranest
import ultranest.stepsampler
from mpi4py import MPI

from .carma_core import CARMAProcess
from .acvf import CovarianceFunction
from .psd import PowerSpectralDensity
from .core import GaussianProcess
from .psdtoacv import PSDToACV
from .utils.psd_utils import get_samples_psd, wrapper_psd_true_samples
from .plots import violin_plots_psd_approx,diagnostics_psd_approx,plot_prior_predictive_PSD,residuals_quantiles,plot_priors_samples
from .utils.gp_utils import tinygp_methods


comm = MPI.COMM_WORLD
rank = comm.Get_rank()

class Inference:
    r"""Class to infer the value of the (hyper)parameters of the Gaussian Process.
    
    
    Attributes
    ----------
    
    process : :class:`~pioran.core.GaussianProcess`
        Gaussian Process object.
    priors: :obj:`function`
        Function to define the priors for the (hyper)parameters.
    method : :obj:`str`
        - "ultranest": nested sampling via ultranest.
    results : :obj:`dict`
        Results of the inference.
    log_dir : :obj:`str`
        Directory to save the results of the inference.
    
    Methods
    -------
    
    run 
        Optimize the (hyper)parameters of the Gaussian Process.
    nested_sampling 
        Optimize the (hyper)parameters of the Gaussian Process using nested sampling via ultranest.
    
    """
    
    def __init__(self, Process: Union[GaussianProcess,CARMAProcess],priors, method :str="ultranest",n_samples=1000,seed_check=0,run_checks=True,log_dir='log_dir'):
        r"""Constructor method for the Optimizer class.

        Instantiate the Inference class.

        Parameters
        ----------
        Process : :class:`~pioran.core.GaussianProcess`
            Process object.
        priors : :obj:`function`
            Function to define the priors for the (hyper)parameters.
        method : :obj:`str`, optional
            "NS": using nested sampling via ultranest
        n_samples : :obj:`int`, optional
            Number of samples to take from the prior distribution, by default 1000
        seed_check : :obj:`int`, optional
            Seed for the random number generator, by default 0    
        run_checks : :obj:`bool`, optional
            Run the prior predictive checks, by default True
        Raises
        ------
        TypeError
            If the method is not a string.
        """
        
        self.process = Process
        self.priors = priors
        self.log_dir = log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        if os.path.isfile(self.log_dir+'/config.json'):
            print("The config file already exists. Loading the previous config.")
            file = open(self.log_dir+'/config.json')
            dict_config_old = json.load(file)
            print('Check the config file:')
            dict_config_new = self.save_config(save_file=False)
            if dict_config_old == dict_config_new:
                print("The config file is the identical.")
            else:
                raise ValueError("The config file is different.")
        else:
            self.save_config()
        
        if isinstance(method, str):
            self.method = method
        else:
            raise TypeError("method must be a string.")  

        #run prior predictive checks
        if run_checks and rank == 0: 
            # MPI.COMM_WORLD.Barrier()
            self.prior_predictive_checks(n_samples,seed_check)
            
            if isinstance(Process, GaussianProcess) and isinstance(self.process.model,PSDToACV):
                if self.process.model.method in tinygp_methods:
                    print(f"The PSD model is a {self.process.model.method} decomposition, checking the approximation.")
                    self.check_approximation(n_samples,seed_check)
        MPI.COMM_WORLD.Barrier()

    def save_config(self,save_file=True):
        """Save the configuration of the inference.
        
        Save the configuration of the inference, process and model in a json file.        
        
        Parameters
        ----------
        save_file : :obj:`bool`, optional
        
        """
        dict_config = {}
        
        # type of process
        if isinstance(self.process,GaussianProcess):
            dict_config['process'] = {'type':'GaussianProcess', 'options':{}}
            dict_config['process']['options']['estimate_mean'] = self.process.estimate_mean
        elif isinstance(self.process,CARMAProcess):
            dict_config['process'] = {'type':'CARMAProcess', 'options':{}}
            dict_config['process']['options']['use_beta'] = self.process.use_beta

        else:
            dict_config['process'] = {'type':'unknown', 'options':{}}
        # 
        # options of the process
        dict_config['process']['options']['estimate_mean'] = self.process.estimate_mean
        dict_config['process']['options']['scale_errors'] = self.process.scale_errors
        dict_config['process']['options']['estimate_variance'] = self.process.estimate_variance
        dict_config['process']['options']['log_transform'] = self.process.log_transform

        # type of model
        if isinstance(self.process.model,PSDToACV):
            dict_config['model'] = {'type':'PowerSpectralDensity','info':{}}
            dict_config['model']['info']['expression'] = self.process.model.PSD.expression
            dict_config['model']['info']['method'] = self.process.model.method
            dict_config['model']['info']['S_low'] = self.process.model.S_low
            dict_config['model']['info']['S_high'] = self.process.model.S_high
            if not ('FFT' in self.process.model.method):
                dict_config['model']['info']['n_components'] = self.process.model.n_components
        
        elif isinstance(self.process.model,CovarianceFunction):
            dict_config['model'] = {'type':'CovarianceFunction','info':{}}
            dict_config['model']['info']['expression'] = self.process.model.expression
        elif isinstance(self.process,CARMAProcess):
            dict_config['model'] = {'type':'CARMA_model','info':{}}
            dict_config['model']['info']['p'] = self.process.p
            dict_config['model']['info']['q'] = self.process.q
            
        # parameters of the model
        dict_config['model']['parameters'] = {}
        dict_config['model']['parameters']['names'] = self.process.model.parameters.names
        dict_config['model']['parameters']['free_parameters'] = self.process.model.parameters.free_names
        
        # data_related
        dict_config['data'] = {}
        dict_config['data']['nb_observation_points'] = len(self.process.observation_indexes)
        dict_config['data']['duration'] = float(self.process.observation_indexes[-1]-self.process.observation_indexes[0])
        dict_config['data']['min_sampling'] = np.diff(self.process.observation_indexes).min()
        dict_config['data']['observation_indexes'] = self.process.observation_indexes.tolist()
        dict_config['data']['observation_values'] = self.process.observation_values.tolist()
        dict_config['data']['observation_errors'] = self.process.observation_errors.tolist()
        
        if save_file:
            with open(self.log_dir+'/config.json', 'w', encoding='utf-8') as f:
                json.dump(dict_config, f, ensure_ascii=False, indent=4)
        
        return dict_config
        
    
 
    def prior_predictive_checks(self,
                                n_samples,
                                seed_check,
                                n_frequencies=1000,
                                plot_prior_samples=True,
                                plot_prior_predictive_distribution=True):
        """Check the prior predictive distribution.
        
        Get samples from the prior distribution and plot them, and calculate the prior predictive
        distribution of the model and plot it. 
        
        Parameters
        ----------
        n_samples : :obj:`int`
            Number of samples to take from the prior distribution, by default 1000
        seed_check : :obj:`int`
            Seed for the random number generator
        plot_prior_samples : :obj:`bool`, optional
            Plot the prior samples, by default True
        plot_prior_predictive_distributions : :obj:`bool`, optional
            Plot the prior predictive distribution of the model, by default True
            
        """
        key = jax.random.PRNGKey(seed_check)
        n_pars = len(self.process.model.parameters.free_names)
        freqs = jnp.geomspace(self.process.model.f0, self.process.model.fN, n_frequencies)


        if self.method == 'ultranest':
            # draw samples from the prior distribution
            uniform_samples = jax.random.uniform(key=key,shape=(n_pars,n_samples))
            self.params_samples = self.priors(np.array(uniform_samples))#[indexes]
            
        else:
            raise NotImplementedError("Only ultranest is implemented for now.")
        
        if plot_prior_samples:
            fig,_ = plot_priors_samples(self.params_samples,self.process.model.parameters.free_names)
            fig.savefig(f"{self.log_dir}/prior_samples.pdf")
            
        if plot_prior_predictive_distribution:
            
            # if model is PSD plot predictive PSD and ACVF
            if isinstance(self.process.model,PSDToACV):
                psd_true_samples = wrapper_psd_true_samples(self.process.model,freqs,self.params_samples.T)
                #
                if not self.process.estimate_variance:
                    fig,_ = plot_prior_predictive_PSD(freqs,psd_true_samples)
                # else:
                    # psd_samples = psd_true_samples/psd_true_samples[...,0,None]
                    # idx_var = self.process.model.parameters.names.index('var')
                    # fig,_ = plot_prior_predictive_PSD(freqs,jnp.transpose(psd_samples.T*self.params_samples[idx_var].flatten()))
                # fig.savefig("prior_predictive_psd_distribution.pdf")
            # psd_samples = wrapper_psd_true_samples(self.process.model,freqs,self.params_samples)
            #     fig,_ = plot_prior_predictive_distribution(params_samples)
            #     fig.savefig("prior_predictive_distribution.pdf")

    def check_approximation(self,n_samples,
                            seed_check,
                            n_frequencies=1000,
                            plot_diagnostics=True,
                            plot_violins=True,
                            plot_quantiles=True):
        
        """Check the approximation of the PSD with the kernel decomposition.
        
        This method will take random samples from the prior distribution and compare the PSD obtained 
        with the SHO decomposition with the true PSD.

        Parameters
        ----------
        n_samples : :obj:`int`
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
            
        """
        freqs = jnp.geomspace(self.process.model.f0, self.process.model.fN, n_frequencies)
        
        # get the true PSD and the SHO PSD samples        
        psd_true,psd_approx = get_samples_psd(self.process.model,freqs,self.params_samples.T)
        # compute the residuals and the ratios
        residuals = psd_true-psd_approx
        ratio = psd_approx/psd_true
        figs = []
        
        if plot_diagnostics:
            fig,_ = diagnostics_psd_approx(f=freqs,
                                            res=residuals,
                                            ratio=ratio,
                                            f_min=self.process.model.f_min_obs,
                                            f_max=self.process.model.f_max_obs)
            fig.savefig(f"{self.log_dir}/diagnostics_psd_approx.pdf")
            figs.append(fig)
        
        if plot_violins:     
            fig,_ = violin_plots_psd_approx(res=residuals,
                                            ratio=ratio)
            fig.savefig(f"{self.log_dir}/violin_plots_psd_approx.pdf")
            figs.append(fig)
        
        if plot_quantiles:
            fig,_ = residuals_quantiles(residuals=residuals,
                                        ratio=ratio,
                                        f=freqs,
                                        f_min=self.process.model.f_min_obs,
                                        f_max=self.process.model.f_max_obs)
            fig.savefig(f"{self.log_dir}/quantiles_psd_approx.pdf")
            figs.append(fig)
            
        return figs,residuals,ratio
                       
    def run(self, verbose=True, **kwargs):
        """ Optimize the (hyper)parameters of the Gaussian Process.
        
        Run the inference method.
        
        Parameters
        ----------
        priors : :obj:`function`, optional
            Function to define the priors for the (hyper)parameters.
        verbose : :obj:`bool`, optional
            Print the results of the optimization, by default False
        **kwargs : :obj:`dict`, optional
            Additional arguments for the optimization method.
                For ML: see 'optimize_ML' docstring
        
        Raises
        ------
        ValueError
            If the method is not "NS".
        
        Returns
        -------
        results: dict
            Results of the optimization. Different keys depending on the method.
        """
        if self.method == "ultranest":
            use_stepsampler = kwargs.pop('use_stepsampler',False)
            if 'user_likelihood' in kwargs:
                print("user_likelihood is used please check the documentation.")
            user_likelihood = kwargs.pop('user_likelihood',self.process.wrapper_log_marginal_likelihood)
            results, sampler = self.nested_sampling(priors=self.priors,user_likelihood=user_likelihood,verbose=verbose,use_stepsampler=use_stepsampler,**kwargs)
        else:
            raise NotImplementedError("Only ultranest is implemented for now.")
        comm.Barrier()
        self.process.model.parameters.set_free_values(results['maximum_likelihood']['point'])#results['posterior']['median'])
        print(self.process.model.parameters.free_values)
        if rank == 0:
            print("\n>>>>>> Plotting corner and trace.")
            sampler.plot()
            print("\n>>>>>> Optimization done.")
            print(self.process)
        return results
        
    def nested_sampling(self,priors,user_likelihood,verbose=True,use_stepsampler=False,**kwargs):
        r""" Optimize the (hyper)parameters of the Gaussian Process with nested sampling via ultranest.

        Perform nested sampling to optimize the (hyper)parameters of the Gaussian Process.    

        Parameters
        ----------
        priors : :obj:`function`
            Function to define the priors for the parameters to be optimized.
        verbose : :obj:`bool`, optional
            Print the results of the optimization and the progress of the sampling, by default True
        **kwargs : :obj:`dict`
            Keyword arguments for ultranest
                - resume: :obj:`bool`
                - log_dir: :obj:`str`
                - run_kwargs: :obj:`dict`
                - Dictionary of arguments for ReactiveNestedSampler.run() see https://johannesbuchner.github.io/UltraNest/ultranest.html#module-ultranest.integrator
        
        Returns
        -------
        results: dict
            Dictionary of results from the nested sampling. 
        """
        
        resume = kwargs.get('resume',True)
        log_dir = kwargs.get('log_dir',self.log_dir)
        run_kwargs = kwargs.get('run_kwargs',{})
        viz = {} if verbose else  {'show_status': False , 'viz_callback': void}
        free_names = self.process.model.parameters.free_names
        slice_steps = kwargs.get('slice_steps',100)
        sampler = ultranest.ReactiveNestedSampler(free_names,user_likelihood ,priors,resume=resume,log_dir=log_dir)
        if use_stepsampler: sampler.stepsampler = ultranest.stepsampler.SliceSampler(nsteps=slice_steps,
                                                generate_direction=ultranest.stepsampler.generate_mixture_random_direction)
        
        if verbose: results = sampler.run(**viz)
        else: results = sampler.run(**run_kwargs, **viz)
        
        return results,sampler
    
def void(*args, **kwargs):
    """ Void function to avoid printing the status of the nested sampling."""
    pass