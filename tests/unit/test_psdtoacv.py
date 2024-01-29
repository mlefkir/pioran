"""Test the PSDToACV class."""

import unittest

import sys

sys.path.append("../../src")

import numpy as np
import jax.numpy as jnp
from pioran.psdtoacv import PSDToACV
from pioran.psd import Lorentzian, OneBendPowerLaw

from celerite2.jax import terms
import celerite2.terms as legacy_terms
from tinygp.kernels.quasisep import Quasisep


T = 1303
dt = 0.0253
S_low = 20
S_high = 10


class TestPSDToACV(unittest.TestCase):
    def test_init_FFT(self):
        psd = Lorentzian([1.2, 4, 5])
        acv = PSDToACV(psd, T=T, dt=dt, S_low=S_low, S_high=S_high, method="FFT")
        self.assertEqual(acv.PSD, psd)

    def test_invalid_PSD(self):
        with self.assertRaises(TypeError):
            PSDToACV("invalid", S_low=20, S_high=10, T=1303, dt=0.0253, method="FFT")

    def test_invalid_dt(self):
        psd = Lorentzian([1.2, 4, 5])
        with self.assertRaises(ValueError):
            PSDToACV(psd, S_low=20, S_high=10, T=0.0253, dt=1303, method="FFT")

    def test_invalid_S_low(self):
        psd = Lorentzian([1.2, 4, 5])
        with self.assertRaises(ValueError):
            PSDToACV(psd, S_low=1, S_high=10, T=1303, dt=0.0253, method="FFT")

    def test_invalid_method(self):
        psd = Lorentzian([1.2, 4, 5])
        with self.assertRaises(ValueError):
            PSDToACV(psd, S_low=20, S_high=10, T=1303, dt=0.0253, method="invalid")
            
    def test_invalid_n_components(self):
        psd = Lorentzian([1.2, 4, 5])
        with self.assertRaises(ValueError):
            PSDToACV(
                psd,
                S_low=20,
                S_high=10,
                T=1303,
                dt=0.0253,
                method="SHO",
                n_components=0,
            )

    def test_missing_values(self):
        psd = OneBendPowerLaw([1, -0.3, 1e-3, -2.5])
        with self.assertRaises(ValueError):
            PSDToACV(
                psd,
                S_low=20,
                S_high=10,
                T=1303,
                dt=0.0253,
                method="SHO",
            )


    def test_spectralmatrix(self):
        """Test the spectral matrix is computed correctly."""
        psd = OneBendPowerLaw([1, -0.3, 1e-3, -2.5])
        acvf = PSDToACV(
            psd,
            S_low=20,
            S_high=10,
            T=1303,
            dt=0.0253,
            method="SHO",
            n_components=20,
        )
        self.assertEqual(acvf.n_components, 20)
        self.assertEqual(jnp.shape(acvf.spectral_matrix), (20, 20))

    def test_type_acvf_tinygp(self):
        """Test the ACVF is a tinygp kernel."""
        psd = OneBendPowerLaw([1, -0.3, 1e-3, -2.5])
        acvf = PSDToACV(
            psd,
            S_low=20,
            S_high=10,
            T=1303,
            dt=0.0253,
            method="SHO",
            n_components=20,
        )
        kernel = acvf.ACVF
        self.assertIsInstance(kernel, Quasisep)

    def test_type_acvf_celerite2(self):
        """Test the ACVF is a celerite2 kernel."""
        psd = OneBendPowerLaw([1, -0.3, 1e-3, -2.5])
        acvf = PSDToACV(
            psd,
            S_low=20,
            S_high=10,
            T=1303,
            dt=0.0253,
            method="SHO",
            use_celerite=True,
            use_legacy_celerite=True,
            n_components=20,
        )
        kernel = acvf.ACVF
        self.assertIsInstance(kernel, legacy_terms.Term)

    def test_type_acvf_celerite2jax(self):
        """Test the ACVF is a celerite2-jax kernel."""
        psd = OneBendPowerLaw([1, -0.3, 1e-3, -2.5])
        acvf = PSDToACV(
            psd,
            S_low=20,
            S_high=10,
            T=1303,
            dt=0.0253,
            method="SHO",
            n_components=20,
            use_celerite=True,
        )
        kernel = acvf.ACVF
        self.assertIsInstance(kernel, terms.Term)

if __name__ == "__main__":
    unittest.main()
