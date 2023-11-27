## progress bar obtained from numpyro source code
## https://github.com/pyro-ppl/numpyro/blob/6a0856b7cda82fc255e23adc797bb79f5b7fc904/numpyro/util.py#L176
## and modified to work with scan using https://www.jeremiecoullon.com/2021/01/29/jax_progress_bar/

import re

import numpy as np
from jax import lax
from jax.experimental import host_callback

_CHAIN_RE = re.compile(r"\d+$")  # e.g. get '3' from 'TFRT_CPU_3'


def save_sampling_results(info:dict,
                          warmup:dict,
                          samples:np.ndarray,
                          log_prob:np.ndarray,
                          log_densitygrad:np.ndarray,
                          filename:str):
    """Save the results of the Monte Carlo runs in a ASDF file.
    
    This file contains the following data:
    - info: a dictionary containing the information about the run, namely:
        - num_params: the number of parameters
        - num_samples: the number of samples
        - num_warmup: the number of warmup samples
        - num_chains: the number of chains
        - ESS: the effective sample size
        - Rhat-split: the split Rhat statistic
    - warmup: a numpy array containing the warmup samples
    - samples: a numpy array containing the samples
    - log_prob: a numpy array containing the log probabilities of the samples
    
    Parameters
    ----------
    info: :obj:`dict`
        A dictionary containing the information about the run
    warmup: :obj:`dict`
        A numpy array containing the warmup samples
    samples: :obj:`jax.Array`
        A numpy array containing the samples
    log_prob: :obj:`jax.Array`
        A numpy array containing the log probabilities of the samples
    log_densitygrad: :obj:`jax.Array`
        A numpy array containing the log density gradients of the samples
    filename: :obj:`str`
        The name of the file to save the data to
    
    """
    import asdf

    tree = {
        'info': info,
        'warmup': warmup,
        "samples": {f'chain_{i}': np.array(samples[i]) for i in range(samples.shape[0])},
        "log_prob": {f'chain_{i}': np.array(log_prob[i]) for i in range(log_prob.shape[0])},
        "log_densitygrad": {f'chain_{i}': np.array(log_densitygrad[i]) for i in range(log_densitygrad.shape[0])},
    }

    # Create the ASDF file object from our data tree
    af = asdf.AsdfFile(tree)

    # Write the data to a new file
    af.write_to(f"{filename}.asdf",all_array_compression="zlib")
    
def progress_bar_factory(num_samples:int, num_chains:int):
    """Factory that builds a progress bar decorator along
    with the `set_tqdm_description` and `close_tqdm` functions
    
    progress bar obtained from numpyro source code
    https://github.com/pyro-ppl/numpyro/blob/6a0856b7cda82fc255e23adc797bb79f5b7fc904/numpyro/util.py#L176
    and modified to work with scan using https://www.jeremiecoullon.com/2021/01/29/jax_progress_bar/

    Parameters
    ----------
    num_samples: :obj:`int`
        The number of samples
    num_chains: :obj:`int`
        The number of chains
    """
    from tqdm.auto import tqdm as tqdm_auto

    if num_samples > 20:
        print_rate = int(num_samples / 20)
    else:
        print_rate = 1

    remainder = num_samples % print_rate

    tqdm_bars = {}
    finished_chains = []
    for chain in range(num_chains):
        tqdm_bars[chain] = tqdm_auto(range(num_samples), position=chain)
        tqdm_bars[chain].set_description("Compiling.. ", refresh=True)

    def _update_tqdm(arg, transform, device):
        chain_match = _CHAIN_RE.search(str(device))
        assert chain_match
        chain = int(chain_match.group())
        tqdm_bars[chain].set_description(f"Running chain {chain}", refresh=False)
        tqdm_bars[chain].update(arg)

    def _close_tqdm(arg, transform, device):
        chain_match = _CHAIN_RE.search(str(device))
        assert chain_match
        chain = int(chain_match.group())
        tqdm_bars[chain].update(arg)
        finished_chains.append(chain)
        if len(finished_chains) == num_chains:
            for chain in range(num_chains):
                tqdm_bars[chain].close()

    def _update_progress_bar(iter_num):
        """Updates tqdm progress bar of a JAX loop only if the iteration number is a multiple of the print_rate
        Usage: carry = progress_bar((iter_num, print_rate), carry)
        """

        _ = lax.cond(
            iter_num == 1,
            lambda _: host_callback.id_tap(
                _update_tqdm, 0, result=iter_num, tap_with_device=True
            ),
            lambda _: iter_num,
            operand=None,
        )
        _ = lax.cond(
            iter_num % print_rate == 0,
            lambda _: host_callback.id_tap(
                _update_tqdm, print_rate, result=iter_num, tap_with_device=True
            ),
            lambda _: iter_num,
            operand=None,
        )
        _ = lax.cond(
            iter_num == num_samples,
            lambda _: host_callback.id_tap(
                _close_tqdm, remainder, result=iter_num, tap_with_device=True
            ),
            lambda _: iter_num,
            operand=None,
        )


    def progress_bar_scan(func):
        """Decorator that adds a progress bar to `body_fun` used in `lax.scan`.
        Note that `body_fun` must either be looping over `np.arange(num_samples)`,
        or be looping over a tuple who's first element is `np.arange(num_samples)`
        This means that `iter_num` is the current iteration number
        """

        def wrapper_progress_bar(carry, x):
            if type(x) is tuple:
                iter_num, *_ = x
            else:
                iter_num = x   
            _update_progress_bar(iter_num)
            result = func(carry, x)
            return result

        return wrapper_progress_bar
    return progress_bar_scan

