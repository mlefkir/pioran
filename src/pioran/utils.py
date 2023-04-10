from jax.numpy import linalg as la
from scipy.spatial.distance import cdist
import jax.numpy as jnp
from jax import jit
import numpy as np

@jit
def EuclideanDistance(xq, xp):
    """Compute the Euclidean distance between two arrays.


    Parameters
    ----------
    xq : array 
        First array of shape (n, 1)
    xp : array 
        Second array of shape (m, 1)

    Returns
    -------
    array of shape (n, m)
    """
    #return cdist(xq, xp, metric='euclidean')
    return jnp.sqrt((xq - xp.T)**2)

# ---- Code from Ahmed Fasih ---- https://gist.github.com/fasiha/fdb5cec2054e6f1c6ae35476045a0bbd
def nearest_positive_definite(A):
    """Find the nearest positive-definite : to input
    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].
    
    Parameters
    ----------
    A : array
        
    
    Returns
    -------
    array
        Nearest positive-definite : to A.  
    
    Notes
    -----
    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    :" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = la.svd(B)

    H = jnp.dot(V.T, jnp.dot(jnp.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(la.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept :es with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random :es of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = jnp.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = jnp.min(jnp.real(la.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3

def decompose_triangular_matrix(M):
    """Decompose a triangular matrix into a vector of unique values.
    
    Decompose a triangular matrix into a vector of unique values and returns the
    indexes to reconstruct the original matrix.

    Parameters
    ----------
    M : (n,n) :obj:`jnp.ndarray`
        Triangular matrix of shape (n,n).

    Returns
    -------
    unique : :obj:`jnp.ndarray`
        Vector of unique values.
    reverse_indexes : :obj:`jnp.ndarray`
        Indexes to reconstruct the original matrix.
    tril_indexes : :obj:`jnp.ndarray`
        Indexes of the lower triangular matrix.
    n : :obj:`int`
        Size of the original matrix.        
    """
    
    n = M.shape[0]
    tril_indexes = jnp.tril_indices(n, k=0)
    trunc = M[tril_indexes]
    unique, reverse_indexes = jnp.unique(trunc,return_inverse=True,size=len(trunc))
    return unique, reverse_indexes, tril_indexes, n


def reconstruct_triangular_matrix(unique, reverse_indexes, tril_indexes, n):
    """Recompose a triangular matrix from a vector of unique values.
    
    Recompose a triangular matrix from a vector of unique values and the indexes    

    Parameters
    ----------
    unique : :obj:`jnp.ndarray`
        Vector of unique values.
    reverse_indexes : :obj:`jnp.ndarray`
        Indexes to reconstruct the original matrix.
    tril_indexes : :obj:`jnp.ndarray`
        Indexes of the lower triangular matrix.
    n : :obj:`int`
        Size of the original matrix.
        
    Returns
    -------
    M : :obj:`jnp.ndarray`
        Triangular matrix of shape (n,n).
    
    Raises
    ------
    ValueError
        If the matrix is not triangular.
    """

    M = jnp.zeros((n,n))
    M = M.at[tril_indexes].set(unique[reverse_indexes])
    return M+M.T-jnp.diag(jnp.diag(M))

def isPD(B):
    """Returns true when input is positive-definite, via Cholesky
    
    Parameters
    ----------
    B : array
        : to test.
        
    Returns
    -------
    bool
        True if B is positive-definite. False otherwise.
    """
    try:
        _ = la.cholesky(B)
        return True
    except la.LinAlgError:
        return False
