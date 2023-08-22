import jax
import jax.numpy as jnp

def SHO_power_spectrum(f,A,f0)->jax.Array:
    """Power spectrum of a simple harmonic oscillator.
    
    .. math:: :label: lorentzianpsd 
    
       \mathcal{P}(f) = \dfrac{A}{1 + (f-f_0)^4}.

    with the amplitude :math:`A`, the position :math:`f_0\ge 0`.
    

    Parameters
    ----------
    f : :obj:`jax.Array`
        Frequency array.
    A : :obj:`float`
        Amplitude of the model.
    f0 : :obj:`float`
        Position of the model.

    Returns
    -------
    :obj:`jax.Array`
    """
    P = A / ( 1 + jnp.power((f/f0),4))
    
    return P