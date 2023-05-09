import jax
import jax.numpy as jnp


def quad_to_roots(quad: jax.Array) -> jax.Array:
    """Convert the coefficients of the quadratic form to coefficients of the AR polynomial."""
    p = quad.shape[0]
    roots = jnp.array([],dtype=jnp.complex128)
    for i in range(int(p/2)):
        c = quad[2*i]
        b = quad[2*i+1]
        delta = b**2 - 4*c
        roots = jax.lax.cond(delta > 0,
            lambda delta: jnp.append(roots,jnp.array([.5*(-b - jnp.sqrt(delta)),.5*(-b + jnp.sqrt(delta))])),
            lambda delta: jnp.append(roots,jnp.array([.5*(-b - 1j*jnp.sqrt(-delta)),.5*(-b + 1j*jnp.sqrt(-delta))])),
            operand=delta)
    if p%2 == 1:
        roots = jnp.append(roots,-quad[-1])
    return roots

def roots_to_quad(roots: jax.Array) -> jax.Array:
    """Convert the roots of the AR polynomial to the coefficients quad of the quadratic polynomial."""
    p = roots.shape[0]
    roots = roots
    quad = jnp.array([])
    for i in range(int(p/2)):
        quad = jnp.append(quad,jnp.array([jnp.abs(roots[2*i])**2,-2*roots[2*i].real]))
    if p%2==1:
        quad = jnp.append(quad,-roots[-1].real)
    return quad

def quad_to_coeff(quad: jax.Array) -> jax.Array:
    """Convert the coefficients quad of the quadratic polynomial to the coefficients alpha of the AR polynomial
    
    Parameters
    ----------
    quad : :obj:`jax.Array`
        Coefficients quad of the quadratic polynomial
    
    Returns
    -------
    :obj:`jax.Array`
        Coefficients alpha of the AR polynomial
    """
    p = quad.shape[0]
    n_products = jnp.ceil(p/2).astype(int)
    alpha = jnp.array([1, quad[1], quad[0]])
    for k in range(2,n_products):
        alpha = jnp.polymul(alpha, jnp.array([1, quad[k+1], quad[k]]), trim_leading_zeros=True)
    if p%2==1:
        alpha = jnp.polymul(alpha, jnp.array([1, quad[-1]]), trim_leading_zeros=True)
    else:
        alpha = jnp.polymul(alpha, jnp.array([1, quad[-1], quad[-2]]), trim_leading_zeros=True)
    return alpha

def lorentzians_to_roots(Widths: jax.Array, Centroids: jax.Array) -> jax.Array:
    """Convert the widths and centroids of the Lorentzian functions to the roots of the AR polynomial."""
    N = len(Widths)
    _Centroids = jnp.insert(Centroids,jnp.arange(1,N+1),-jnp.roll(Centroids,N))
    Combined_Centroids = jnp.delete(_Centroids,jnp.where(_Centroids==0)[0][1::2])
    Combined_Widths = jnp.repeat(Widths,2)
    Combined_Widths = jnp.delete(Combined_Widths,jnp.where(_Centroids==0)[0][1::2])
    roots = -2*jnp.pi * ( Combined_Widths + 1j*Combined_Centroids )
    return roots

def get_U(roots_AR: jax.Array) -> jax.Array:
    U = jnp.ones(len(roots_AR))
    for k in range(1,len(roots_AR)):
        U = jnp.vstack((U,roots_AR**k))
    return U

def get_V(J: jax.Array, roots_AR: jax.Array) -> jax.Array:
    p = len(roots_AR)
    arr = jnp.zeros((p,p),dtype=jnp.complex128)
    arr = arr.at[:].set(roots_AR)
    den = arr + jnp.conj(arr.T)
    num = -J.reshape(1,p).T*jnp.conj(J).T
    V = jnp.conj(num/den.T)
    return V.T