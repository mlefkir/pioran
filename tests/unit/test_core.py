"""unit tests for core.py module
"""

import unittest

import sys
sys.path.append('../../src')

import jax
import jax.numpy as jnp
from pioran.acvf_base import CovarianceFunction
from pioran.core import GaussianProcess


key = jax.random.PRNGKey(0)
x = jnp.linspace(0,10,100)
y = jnp.sin(x)+jax.random.normal(key, shape=x.shape)*0.1


