import jax
import jax.numpy as jnp

def SHO_power_spectrum(f,A,f0)->jax.Array:
    r"""Power spectrum of a stochastic harmonic oscillator.
    
    .. math:: :label: sho_power_spectrum 
    
       \mathcal{P}(f) = \dfrac{A}{1 + (f-f_0)^4}.

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
    P = A / ( 1 + jnp.power((f/f0),4))
    
    return P

def SHO_autocovariance(tau,A,f0):
    r"""Autocovariance function of a stochastic harmonic oscillator.
    
    .. math:: :label: sho_autocovariance
    
       \mathcal{R}(\tau) = A \times 2\pi f_0 \exp\left(-\dfrac{ 2\pi f_0 \tau}{\sqrt{2}}\right) \cos\left(\dfrac{ 2\pi f_0 \tau}{\sqrt{2}}-\dfrac{\pi}{4}\right).
       
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
    return A*(2*jnp.pi*f0)*jnp.exp(-1/jnp.sqrt(2)*2*jnp.pi*f0*tau)*\
            jnp.cos (2*jnp.pi*f0*tau / jnp.sqrt(2)-jnp.pi/4)