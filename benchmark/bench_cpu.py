import sys
import timeit
import jax

jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'cpu')

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

from equinox import filter_jit
from pioran.psd import OneBendPowerLaw
from pioran import GaussianProcess

p = [5,-.5,.014,-3.21]

psd = OneBendPowerLaw(p)
t,y,yerr = np.genfromtxt("simulate_long.txt").T

n_samples = [50, 100, 200, 500, 800, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000]
n_components = [10, 20, 25, 30, 40, 50]


full_mean = []

for J in n_components:
    mean = []
    print(f'J={J}\t\n')
    for i,N in enumerate(n_samples):
        sys.stdout.write('\r'+str(i)+'/'+str(len(n_samples)))
        t_trunc, y_trunc, yerr_trunc = t[:N], y[:N], yerr[:N]
        psd = OneBendPowerLaw(p)
        gp = GaussianProcess(psd,t_trunc,y_trunc,yerr_trunc,method='SHO',use_tinygp=True,n_components=J)
        fun = lambda: (filter_jit(gp.wrapper_log_marginal_likelihood)([-.5,.014,-2.21,1,1.,.5])).block_until_ready()
        l = fun()
        res = timeit.Timer(fun).autorange()
        mean.append(res[1]/res[0])
    full_mean.append(mean)

full_mean = np.array(full_mean)
data = np.vstack((np.array(n_components),full_mean.T))

np.savetxt(f'benchmark_cpu.txt',data,header="Benchmark CPU\nn_components\ntime(s)")

