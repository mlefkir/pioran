"""Utility functions for PSD models."""
import jax
import jax.numpy as jnp

from ..psdtoacv import PSDToACV


def SHO_power_spectrum(f: jax.Array, A: float, f0: float) -> jax.Array:
    r"""Power spectrum of a stochastic harmonic oscillator.

    .. math:: :label: sho_power_spectrum

       \mathcal{P}(f) = \dfrac{A}{1 + (f/f_0)^4}.

    with the amplitude :math:`A`, the position :math:`f_0\ge 0`.


    Parameters
    ----------
    f : :obj:`jax.Array`
        Frequency array.
    A : :obj:`float`
        Amplitude.
    f0 : :obj:`float`
        Position.

    Returns
    -------
    :obj:`jax.Array`
    """
    P = A / (1 + jnp.power((f / f0), 4))

    return P

def DRWCelerite_power_spectrum(f: jax.Array, A: float, f0: float) -> jax.Array:
    r"""Power spectrum of the DRW+Celerite component.

    .. math:: :label: drwcel_power_spectrum

       \mathcal{P}(f) = \dfrac{A}{1 + (f/f_0)^6}.

    with the amplitude :math:`A`, the position :math:`f_0\ge 0`.


    Parameters
    ----------
    f : :obj:`jax.Array`
        Frequency array.
    A : :obj:`float`
        Amplitude.
    f0 : :obj:`float`
        Position.

    Returns
    -------
    :obj:`jax.Array`
    """
    P = A / (1 + jnp.power((f / f0), 6))

    return P


def SHO_autocovariance(tau: jax.Array, A: float, f0: float) -> jax.Array:
    r"""Autocovariance function of a stochastic harmonic oscillator.

    .. math:: :label: sho_autocovariance

       K(\tau) = A \times 2\pi f_0 \exp\left(-\dfrac{ 2\pi f_0 \tau}{\sqrt{2}}\right) \cos\left(\dfrac{ 2\pi f_0 \tau}{\sqrt{2}}-\dfrac{\pi}{4}\right).

    with the amplitude :math:`A`, the position :math:`f_0\ge 0`.


    Parameters
    ----------
    tau : :obj:`jax.Array`
        Time lag array.
    A : :obj:`float`
        Amplitude.
    f0 : :obj:`float`
        Position.

    Returns
    -------
    :obj:`jax.Array`
    """
    return (
        A
        * (2 * jnp.pi * f0)
        * jnp.exp(-1 / jnp.sqrt(2) * 2 * jnp.pi * f0 * tau)
        * jnp.cos(2 * jnp.pi * f0 * tau / jnp.sqrt(2) - jnp.pi / 4)
    )


def get_psd_approx_samples(
    psd_acvf: PSDToACV, f: jax.Array, params_samples: jax.Array
) -> jax.Array:
    """Get the true PSD model and the approximated PSD using SHO decomposition.

    Given a PSDToACV object and a set of parameters, return the true PSD and the approximated PSD using SHO decomposition.

    Parameters
    ----------
    psd_acvf : :class:`~pioran.psdtoacv.PSDToACV`
        PSDToACV object.
    f : :obj:`jax.Array`
        Frequency array.
    params_samples : :obj:`jax.Array`
        Parameters of the PSD model.

    Returns
    -------
    :obj:`jax.Array`
        True PSD.
    :obj:`jax.Array`
        Approximated PSD.
    """

    psd_acvf.parameters.set_free_values(params_samples)
    if psd_acvf.method == "SHO":
        a, f_c = psd_acvf.get_approx_coefs()
        psd_SHO = SHO_power_spectrum(f, a[..., None], f_c[..., None]).sum(axis=0)
        psd_model = psd_acvf.PSD.calculate(f)
        psd_model /= psd_model[..., 0, None]
        return psd_model, psd_SHO
    else:
        raise NotImplementedError("Only SHO is implemented for now.")


def get_samples_psd(
    psd_acvf: PSDToACV, f: jax.Array, params_samples: jax.Array
) -> jax.Array:
    """Just a wrapper for jax.vmap(get_psd_approx_samples,(None,None,0))(psd_acvf,f,params_samples)

    Parameters
    ----------
    psd_acvf : :class:`~pioran.psdtoacv.PSDToACV`
        PSDToACV object.
    f : :obj:`jax.Array`
        Frequency array.
    params_samples : :obj:`jax.Array`
        Parameters of the PSD model.
    """
    return jax.vmap(get_psd_approx_samples, (None, None, 0))(
        psd_acvf, f, params_samples
    )


def get_psd_true_samples(
    psd_acvf: PSDToACV, f: jax.Array, params_samples: jax.Array
) -> jax.Array:
    """Get the true PSD model.

    Parameters
    ----------
    psd_acvf : :class:`~pioran.psdtoacv.PSDToACV`
        PSDToACV object.
    f : :obj:`jax.Array`
        Frequency array.
    params_samples : :obj:`jax.Array`
        Parameters of the PSD model.
    """
    psd_acvf.parameters.set_free_values(params_samples)
    psd_model = psd_acvf.PSD.calculate(f)
    return psd_model


def wrapper_psd_true_samples(
    psd_acvf: PSDToACV, f: jax.Array, params_samples: jax.Array
) -> jax.Array:
    """Just a wrapper for jax.vmap(get_psd_true_samples,(None,None,0))(psd_acvf,f,params_samples)

    Parameters
    ----------
    psd_acvf : :class:`~pioran.psdtoacv.PSDToACV`
        PSDToACV object.
    f : :obj:`jax.Array`
        Frequency array.
    params_samples : :obj:`jax.Array`
        Parameters of the PSD model.
    """
    return jax.vmap(get_psd_true_samples, (None, None, 0))(psd_acvf, f, params_samples)
