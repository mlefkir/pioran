import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from mpi4py import MPI

from pioran.core import GaussianProcess
from pioran.diagnostics import Visualisations
from pioran.inference import Inference
from pioran.psd import OneBendingPowerLaw

plt.style.use("https://github.com/mlefkir/beauxgraphs/raw/main/beautifulgraphs_colblind.mplstyle")


comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# load data
hdu = fits.open("F9intensiveLCs.fits")

flux = np.array(hdu["W2"].data["Flux"],dtype=float)#[:30]
flux_err = np.array(hdu["W2"].data["Error"],dtype=float)#[:30]
mjd = np.array(hdu["W2"].data["MJD"],dtype=float)#[:30]
t = mjd - mjd[0]

# define constants
abs_mean = jnp.abs(jnp.mean(flux))
f_min = 1/(t[-1]-t[0])
f_max = 1/(2*np.min(np.diff(t)))
var = jnp.var(flux)
low, high = jnp.log10(f_min/10), jnp.log10(f_max*10)

#model 
psd_model = OneBendingPowerLaw([1,1,1])
# Gaussian process
gp = GaussianProcess(psd_model,t,flux,flux_err,S_low=10,S_high=10)

name = 'onebendPL'

# priors
def priors(cube):
    params = cube.copy()
    params[0] = cube[0]*(5-(-5))+(-5) # index_1
    params[1] = 10**(cube[1]*(high-low)+low) # f_1
    params[2] = cube[2]*10-5 # delindex_2
    params[3] = 10** ( cube[3] * 4  -2) # var
    params[4] = cube[4]*2.5+.5  # nu
    params[5] = cube[5]*abs_mean*5 # mu
    return params

# inference object
Inf = Inference(gp,method="NS")
res = Inf.run(priors,log_dir=name) # to speed up inference use: ,use_stepsampler=1,slice_steps=80)#,user_likelihood=likelihood)

# do some plotting after inference
comm.Barrier()
if rank == 0 :
    Vis = Visualisations(gp,filename=f"{name}/plots/{name}")
    Vis.plot_timeseries_diagnostics()
    Vis.posterior_predictive_checks(res['samples'],plot_ACVF=False)
    