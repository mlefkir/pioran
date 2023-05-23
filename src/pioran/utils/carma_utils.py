import jax
import jax.numpy as jnp
from ..tools import Array_type


def initialise_CARMA_object(self, p, q, AR_quad=None,MA_quad=None, beta=None,use_beta=True, lorentzian_centroids=None, lorentzian_widths=None,weights=None,**kwargs) -> None:
    
        # if we provide the quadratic coefficients 
        if AR_quad is not None:
            
            # set the AR parameters
            if isinstance(AR_quad,Array_type):
                assert len(AR_quad) == self.p, "AR_quad must have length p"
                for i,ar in enumerate(AR_quad):
                    self.parameters.append(f"a_{i+1}",ar,True,hyperparameter=True)
            else:
                assert self.p == 1, "p must be 1 if AR_quad is not an array"
                self.parameters.append(f"a_1",AR_quad,True,hyperparameter=True)
            
            # set the MA parameters
            
            if self.q == 0:
                assert beta is None and MA_quad is None, "beta must be None if q = 0"
                self.parameters.append(f"beta_{0}",1,False,hyperparameter=True)

            if self.q > 0:
                if beta is None and self.use_beta:
                    raise ValueError("beta is required if q >= 1")
                elif MA_quad is None and not self.use_beta:
                    raise ValueError("MA_quad is required if q >= 1")
                
                if self.use_beta:
                    self.parameters.append(f"beta_{0}",1,False,hyperparameter=True)
                    assert len(beta) == self.q, "weights must have length q"
                    for i,ma in enumerate(beta):
                        self.parameters.append(f"beta_{i+1}",float(ma),True,hyperparameter=True)
                else:
                    assert len(MA_quad) == self.q, "MA_quad must have length q"
                    for i,ma in enumerate(MA_quad):
                        self.parameters.append(f"b_{i+1}",float(ma),True,hyperparameter=True)
            if self.use_beta:
                for i in range(self.q,self.p-1):
                    self.parameters.append(f"beta_{i+1}",float(0.),False,hyperparameter=True)
                
        elif lorentzian_centroids is not None and lorentzian_widths is not None :
             
            assert len(lorentzian_centroids) == len(lorentzian_widths), "lorentzian_centroids and lorentzian_widths must have the same length"
            if self.p % 2 == 0:
                assert jnp.count_nonzero(lorentzian_centroids) == len(lorentzian_centroids), "When p is even, lorentzian_centroids must have non-zero elements"
                assert len(lorentzian_centroids) == self.p//2, "lorentzian_centroids must have p//2 non-zero elements"
            else:
                assert jnp.count_nonzero(lorentzian_centroids)+1 == len(lorentzian_centroids), "When p is odd, lorentzian_centroids must have p//2+1 non-zero elements"
                assert jnp.count_nonzero(lorentzian_centroids) == (self.p-1)//2, "lorentzian_centroids must have p//2+1 non-zero elements"

            roots = lorentzians_to_roots(lorentzian_widths,lorentzian_centroids)
            AR_quad = roots_to_quad(roots)
            for i,ar in enumerate(AR_quad):
                    self.parameters.append(f"a_{i+1}",float(ar),True,hyperparameter=True)
            self.parameters.append("beta_0",float(1.),False,hyperparameter=True)
            
            if self.q == 0:
                assert weights is None, "weights must be None if q = 0"
            else:
                assert len(weights) == self.q, "weights must have length q"
                for i,ma in enumerate(weights):
                        self.parameters.append(f"beta_{i+1}",float(ma),True,hyperparameter=True)
            for i in range(self.q,self.p-1):
                self.parameters.append(f"beta_{i+1}",float(0.),False,hyperparameter=True)
        else:
            raise ValueError("Either AR_roots and MA_roots or AR_quad and MA_quad or lorentzian_centroids, lorentzian_widths and weights must be provided")

    
    

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

def MA_quad_to_coeff(q,quad: jax.Array) -> jax.Array:
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
    MA_roots = quad_to_roots(quad)
    beta = jnp.poly(MA_roots)
    # code below does not work with JIT as it uses indexes
    # n_products = jnp.ceil(q/2).astype(int)
    # beta = jnp.array([1, quad[1], quad[0]])
    # print(n_products)
    # if q >2:
    #     for k in range(2,n_products):
    #         beta = jnp.polymul(beta, jnp.array([1, quad[k+1], quad[k]]), trim_leading_zeros=True)
    #     if q%2==1:
    #         beta = jnp.polymul(beta, jnp.array([1, quad[-1]]), trim_leading_zeros=True)
    #     else:
    #         beta = jnp.polymul(beta, jnp.array([1, quad[-1], quad[-2]]), trim_leading_zeros=True)
    return (beta/beta[-1])[::-1]

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