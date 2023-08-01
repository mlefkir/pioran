import jax.numpy as jnp

def SHO_power_spectrum(f,A,f0):
    P = A / ( 1 + jnp.power((f/f0),4))
    
    return P