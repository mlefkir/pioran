# Pioran

Power spectrum Inference Of RANdom time series
<!-- Gaussian Processes Regression on irregularly sampled time series  -->

## Installation

```bash
pip install https://github.com/mlefkir/pioran.git
```

## TODO

- [ ] Add a proper README
- [ ] Add a proper documentation
- [ ] Add a proper example
- [ ] Add a proper test suite
- [ ] Add a proper CI

## TODO - more

- [ ] Operator overloading for `+`, `-`, `*`, `/` and `**` for parameters
- [ ] Add multi-dimensional input support
- [ ] connect inference pipeline

```python
import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import lognorm
from pioran import GaussianProcess, Inference
from pioran.psd import OneBendPowerLaw
from pioran.diagnostics import Visualisations
from mpi4py import MPI
from get_priors_mean_var import get_priors

comm = MPI.COMM_WORLD
size = comm.Get_size()

filename = sys.argv[1]

log_dir = 
t, y, yerr = np.loadtxt(f"{filename}", unpack=True)

psd = OneBendPowerLaw([1, 1, 1, 1])
gp = GaussianProcess(
    psd,
    t,
    y,
    yerr,
    S_low=100,
    S_high=20,
    use_tinygp=True,
    n_components=20,
    method="SHO")

min_index_1, max_index_1 = -2, 0.25
min_f_1, max_f_1 = gp.model.f0 * 10, gp.model.fN / 10
min_index_2, max_index_2 = -3.9, -0.5
log10_min_c = -7  # log10min value for const


def priors(cube):
    params = cube.copy()
    params[0] = cube[0] * (max_index_1 - min_index_1) + min_index_1  # alpha_1
    params[1] = 10 ** (
        cube[1] * (np.log10(max_f_1) - np.log10(min_f_1)) + np.log10(min_f_1)
    )  # f_1
    params[2] = cube[2] * (max_index_2 - min_index_2) + min_index_2  # alpha_2
    params[3] = lognorm.ppf(cube[3], 1.25, loc=0, scale=0.5)  # var
    params[4] = cube[4] * 4.9 + 0.1  # nu
    params[5] = lognorm.ppf(cube[5], 1, loc=0, scale=3)  # mu
    return params

inf = Inference(gp, priors, method="ultranest", run_checks=False, log_dir=log_dir)
res = inf.run()

comm.Barrier()
if rank == 0:
    samples = res["samples"]
    vis = Visualisations(gp, filename=f"{inf.log_dir}/visualisations")
    vis.plot_timeseries_diagnostics_old()
    vis.posterior_predictive_checks(samples,plot_PSD=True,plot_ACVF=False)
    #vis.plot_timeseries_diagnostics(samples)

``