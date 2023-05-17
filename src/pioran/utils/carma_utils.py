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

def PowerSpectrum(f: jax.Array,alpha,beta,sigma) -> jax.Array:
    r"""Computes the power spectrum of the CARMA process.

    Parameters
    ----------
    f : :obj:`jax.Array`
        Frequencies at which the power spectrum is evaluated.
    alpha : :obj:`jax.Array`
        Coefficients of the AR polynomial.
    beta : :obj:`jax.Array`
        Coefficients of the MA polynomial.
    sigma : :obj:`float`
        Standard deviation of the white noise process.
    
    Returns
    -------
    P : :obj:`jax.Array`
        Power spectrum of the CARMA process.
    
    """

    num = jnp.polyval(beta[::-1],2j*jnp.pi*f)
    den = jnp.polyval(alpha,2j*jnp.pi*f)
    P = (sigma  * jnp.abs(num)**2 /jnp.abs(den)**2).flatten()
    return P

def Autocovariance(tau,roots_AR,beta,sigma):
    Frac = 0
    q = beta.shape[0]
    for k, r in enumerate(roots_AR):
        A, B = 0, 0
        for l in range(q):
            A += beta[l]*r**l
            B += beta[l]*(-r)**l
        Den = -2*jnp.real(r)
        for l, root_AR_bis in enumerate(jnp.delete(roots_AR,k)):
            Den *= (root_AR_bis - r)*(jnp.conjugate(root_AR_bis) + r)
        Frac += A*B/Den*jnp.exp(r*tau)
    return sigma**2 * Frac.real