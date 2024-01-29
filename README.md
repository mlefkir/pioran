# Pioran

Power spectrum Inference Of RANdom time series
<!-- Gaussian Processes Regression on irregularly sampled time series  -->

## Installation

First clone the repository:

```bash
git clone git@github.com:mlefkir/pioran.git
```

Then install the base package using pip:
```bash
pip install .
```
If you need the celerite2 backend for the Gaussian process modelling, install it using:
```bash
pip install .[celerite2]
```
Note this will compile the celerite2 package from source, which may take a while!

Other backends are available: ``ultranest``, ``blackjax`` for using these tools for inference.

### Example usage

A simple example of how to use the package using ``ultranest`` as the inference tool.

```python
import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
from pioran import GaussianProcess, Inference
from pioran.psd import OneBendPowerLaw
from pioran.diagnostics import Visualisations
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

filename = "lightcurve.txt"
log_dir = "results"
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

```
