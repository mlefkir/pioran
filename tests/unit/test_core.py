"""unit tests for core.py module
"""

import unittest

import sys

sys.path.append("../../src")

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from pioran.psd import OneBendPowerLaw
from pioran.core import GaussianProcess
from pioran.psdtoacv import PSDToACV
import tinygp
import celerite2
import equinox as eqx

key = jax.random.PRNGKey(0)
subkey, key = jax.random.split(key)
x = jnp.linspace(0, 10, 100)
y = jnp.sin(x) + jax.random.normal(key, shape=x.shape) * 0.1
yerr = jnp.abs(jax.random.normal(key, shape=x.shape))


class TestGaussianProcess(unittest.TestCase):
    
    def test_init_noerr(self):
        psd = OneBendPowerLaw([1, 0.2, 1e-2, 3.4])
        gp = GaussianProcess(
            psd,
            x,
            y,
            S_low=100,
            S_high=20,
            use_tinygp=True,
            n_components=20,
            method="SHO",
        )
        self.assertIsInstance(gp.model, PSDToACV)
        self.assertIsInstance(gp.model.ACVF, tinygp.kernels.quasisep.Quasisep)

    def test_init_tinygp(self):
        psd = OneBendPowerLaw([1, 0.2, 1e-2, 3.4])
        gp = GaussianProcess(
            psd,
            x,
            y,
            yerr,
            S_low=100,
            S_high=20,
            use_tinygp=True,
            n_components=20,
            method="SHO",
        )
        self.assertIsInstance(gp.model, PSDToACV)
        self.assertIsInstance(gp.model.ACVF, tinygp.kernels.quasisep.Quasisep)

    def test_init_celerite(self):
        psd = OneBendPowerLaw([1, 0.2, 1e-2, 3.4])
        gp = GaussianProcess(
            psd,
            x,
            y,
            yerr,
            S_low=100,
            S_high=20,
            use_celerite=True,
            n_components=20,
            method="SHO",
        )
        self.assertIsInstance(gp.model, PSDToACV)
        self.assertIsInstance(gp.model.ACVF, celerite2.jax.terms.Term)

    def test_likelihood_celerite(self):
        psd = OneBendPowerLaw([1, 0.2, 1e-2, 3.4])
        gp = GaussianProcess(
            psd,
            x,
            y,
            yerr,
            S_low=100,
            S_high=20,
            use_celerite=True,
            n_components=20,
            method="SHO",
        )
        like_cel = gp.compute_log_marginal_likelihood()
        self.assertTrue(jnp.isfinite(like_cel))

    def test_likelihood_tinygp(self):
        psd = OneBendPowerLaw([1, 0.2, 1e-2, 3.4])
        gp = GaussianProcess(
            psd,
            x,
            y,
            yerr,
            S_low=100,
            S_high=20,
            use_tinygp=True,
            n_components=20,
            method="SHO",
        )
        like_tgp = gp.compute_log_marginal_likelihood()
        self.assertTrue(jnp.isfinite(like_tgp))

    def test_equal_likelihood_tinygp_celerite(self):
        psd_cel = OneBendPowerLaw([1, 0.2, 1e-2, 3.4])
        gp_cel = GaussianProcess(
            psd_cel,
            x,
            y,
            yerr,
            S_low=100,
            S_high=20,
            use_celerite=True,
            n_components=20,
            method="SHO",
        )
        psd_tgp = OneBendPowerLaw([1, 0.2, 1e-2, 3.4])
        gp_tgp = GaussianProcess(
            psd_tgp,
            x,
            y,
            yerr,
            S_low=100,
            S_high=20,
            use_tinygp=True,
            n_components=20,
            method="SHO",
        )
        like_cel = gp_cel.compute_log_marginal_likelihood()
        like_tgp = gp_tgp.compute_log_marginal_likelihood()
        self.assertAlmostEqual(
            like_cel,like_tgp
        )
    
    def test_likelihood_gradient_tinygp(self):
        psd_tgp = OneBendPowerLaw([1, 0.2, 1e-2, 3.4])
        gp_tgp = GaussianProcess(
            psd_tgp,
            x,
            y,
            yerr,
            S_low=100,
            S_high=20,
            use_tinygp=True,
            n_components=20,
            method="SHO",
        )
        p = np.array(gp_tgp.model.parameters.free_values)
        like = gp_tgp.wrapper_log_marginal_likelihood
        grad = eqx.filter_grad(like)
        grad_tgp = grad(p)
        self.assertTrue(jnp.all(jnp.isfinite(grad_tgp)))
        
    def test_likelihood_gradient_celerite(self):
        psd_tgp = OneBendPowerLaw([1, 0.2, 1e-2, 3.4])
        gp_tgp = GaussianProcess(
            psd_tgp,
            x,
            y,
            yerr,
            S_low=100,
            S_high=20,
            use_celerite=True,
            n_components=20,
            method="SHO",
        )
        p = np.array(gp_tgp.model.parameters.free_values)
        like = gp_tgp.wrapper_log_marginal_likelihood
        grad = eqx.filter_grad(like)
        grad_cel = grad(p)
        self.assertFalse(jnp.all(jnp.isfinite(grad_cel)))