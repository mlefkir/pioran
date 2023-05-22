from typing import Union, Tuple, List, Optional

import equinox as eqx
import jax
import jax.numpy as jnp

from .parameters import ParametersModel
from .tools import Array_type
from .utils.carma_utils import (get_U, get_V, lorentzians_to_roots,MA_quad_to_coeff,
                                quad_to_coeff, quad_to_roots, roots_to_quad)


class CARMA_model(eqx.Module):
    r"""Base class for Continuous-time AutoRegressive Moving Average (CARMA) models. Inherits from eqxinox.Module.
    
    This class implements the basic functionality for CARMA models

    Parameters
    ----------
    parameters : :obj:`ParametersModel`
        Parameters of the model.
    p : :obj:`int`
        Order of the AR part of the model.
    q : :obj:`int`
        Order of the MA part of the model. 0 <= q < p
        
    Attributes
    ----------
    parameters : :obj:`ParametersModel`
        Parameters of the model.
    p : :obj:`int`
        Order of the AR part of the model.
    q : :obj:`int`
        Order of the MA part of the model. 0 <= q < p
    _p : :obj:`int`
        Order of the AR part of the model. p+1
    _q : :obj:`int`
        Order of the MA part of the model. q+1   
    
    
    """
    
    parameters: ParametersModel
    ndims: int
    p: int
    q: int
    _p: int
    _q: int
    use_beta: bool
    
    def __init__(self, p, q, AR_quad=None,MA_quad=None, beta=None,use_beta=True, lorentzian_centroids=None, lorentzian_widths=None,weights=None,**kwargs) -> None:
        """Constructor method
        """
        sigma = kwargs.get("sigma",1)
        self.parameters = ParametersModel(["sigma"],[sigma],[True],[True])
        
        self.p = p
        self.q = q
        assert self.q < self.p, "q must be smaller than p"
        
        self._p = p+1
        self._q = q+1
        self.ndims = 1
        self.use_beta = use_beta
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

    def __str__(self) ->str:
        r"""String representation of the model.
        
        Also prints the roots and coefficients of the AR and MA parts of the model.
        
        """
        s = ''
        s += f"CARMA({self.p},{self.q}) model\n"
        s += self.parameters.__str__()
        s += "\n"
        s += f"AR roots: {self.get_AR_roots()}\n"
        s += f"AR coefficients: {self.get_AR_coeffs()}\n"
        return s
    
    def __repr__(self) ->str:
        return self.__str__()
    
    def PowerSpectrum(self,f: jax.Array) -> jax.Array:
        r"""Computes the power spectrum of the CARMA process.
    
        Parameters
        ----------
        f : :obj:`jax.Array`
            Frequencies at which the power spectrum is evaluated.
        
        
        Returns
        -------
        P : :obj:`jax.Array`
            Power spectrum of the CARMA process.
        
        """
        alpha = self.get_AR_coeffs()
        beta = self.get_MA_coeffs() 

        num = jnp.polyval(beta[::-1],2j*jnp.pi*f)
        den = jnp.polyval(alpha,2j*jnp.pi*f)
        P = (self.parameters["sigma"].value  * jnp.abs(num)**2 /jnp.abs(den)**2).flatten()
        return P
    
    def get_AR_quads(self) -> jax.Array:
        r"""Returns the quadratic coefficients of the AR part of the model.

        Iterates over the parameters of the model and returns the quadratic
        coefficients of the AR part of the model.

        Returns
        -------
        :obj:`jax.Array`
            Quadratic coefficients of the AR part of the model.
        """
        return jnp.array([self.parameters[f"a_{i}"].value for i in range(1,self._p)])

    def get_MA_quads(self) -> jax.Array:
        """Returns the quadratic coefficients of the MA part of the model.
        
        Iterates over the parameters of the model and returns the quadratic
        coefficients of the MA part of the model.
        
        Returns
        -------
        :obj:`jax.Array`
            Quadratic coefficients of the MA part of the model.
        """
        return jnp.array([self.parameters[f"b_{i}"].value for i in range(1,self.q+1)])
       
    def get_AR_coeffs(self) -> jax.Array:
        r"""Returns the coefficients of the AR part of the model.
        
        
        Returns
        -------
        :obj:`jax.Array`
            Coefficients of the AR part of the model.
        """
        if self.p == 1:
            alpha = jnp.array([1,self.parameters["a_1"].value])
        elif self.p == 2:
            alpha = jnp.array([1,self.parameters["a_2"].value ,self.parameters["a_1"].value])
        else :
            alpha = quad_to_coeff(self.get_AR_quads())
        return alpha
    
    def get_MA_coeffs(self) -> jax.Array:
        r"""Returns the quadratic coefficients of the AR part of the model.

        Iterates over the parameters of the model and returns the quadratic
        coefficients of the AR part of the model.

        Returns
        -------
        :obj:`jax.Array`
            Quadratic coefficients of the AR part of the model.
        """
        if self.use_beta:
            return jnp.array([self.parameters[f"beta_{i}"].value for i in range(self.p)])
        else:
            return jnp.append(MA_quad_to_coeff(self.q,self.get_MA_quads()),jnp.zeros(self.p-self.q-1))
    
    def get_AR_roots(self) -> jax.Array:
        r"""Returns the roots of the AR part of the model.
        
        Returns
        -------
        :obj:`jax.Array`
            Roots of the AR part of the model.
        """
        return quad_to_roots(self.get_AR_quads())
    
    def Autocovariance(self, tau: jax.Array) -> jax.Array:
        r"""Compute the autocovariance function of a CARMA(p,q) process."""
        Frac = 0
        roots_AR = self.get_AR_roots()
        beta = self.get_MA_coeffs()
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
        return self.parameters["sigma"].value**2 * Frac.real
    
    # @eqx.filter_jit
    def init_statespace(self,y_0=None,errsize=None) -> Tuple[jax.Array,jax.Array] | Tuple[jax.Array,jax.Array,jax.Array,jax.Array,jax.Array]:
        r"""Initialises the state space representation of the model
        
        Parameters
        
        """             
        # CAR(1) process
        if self.p == 1:    
            X = jnp.zeros((1,1))
            P = jnp.atleast_2d([[self.parameters['sigma'].value/ (2*self.parameters['a_1'].value)]])
            return X,P
        
        # CARMA(p,q) process in the rotated basis
        if y_0 is None or errsize is None:
            raise ValueError('y_0 and errsize must be provided')
        
        beta = self.get_MA_coeffs().reshape(1,self.p)
        e = jnp.append(jnp.zeros(self.p-1),self.parameters['sigma'].value).reshape(self.p,1)
        AR_roots = self.get_AR_roots()
        
        U = get_U(AR_roots)
        J = jnp.linalg.solve(U,e)

        V = get_V(J, AR_roots)
        b_rot = ( beta @ U )

        X = jnp.zeros((self.p,1),dtype=jnp.complex128)
        var = jnp.real(b_rot@V@jnp.conj(b_rot.T)) + errsize**2
        
        y_k = jnp.real(b_rot @ X)
        S_k = jnp.real(b_rot @ V @ jnp.conj(b_rot.T)) + errsize**2
            
        K = V @ jnp.conj(b_rot.T) / var
        X += K * y_0
        P = V - var * K @ jnp.conj(K.T)
        

        loglike =  -0.5 * ( jnp.log(S_k) + (y_0 -y_k)**2 / S_k  + jnp.log(2*jnp.pi) )
        
        return X, P, V, b_rot, loglike

    # @eqx.filter_jit
    def statespace_representation(self,dt: jax.Array) -> Tuple[jax.Array,jax.Array,jax.Array] | jax.Array:
        if self.p == 1:
            F = jnp.exp(-self.parameters['a_1'].value*dt)
            Q = jnp.atleast_2d(self.parameters['sigma'].value / (2*self.parameters['a_1'].value)  * (1 - F**2 ))
            H = jnp.ones((self.ndims,1))
            return F.reshape(1,1),Q,H
        
        F = jnp.diag(jnp.exp(self.get_AR_roots()*dt))
        return F
   