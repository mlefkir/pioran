from numpy import linalg as la
from scipy.spatial.distance import cdist
import numpy as np


def EuclideanDistance(xq, xp):
    """Compute the Euclidian distance between two arrays.

    using scipy.spatial.distance.cdist as it seems faster than a homemade version

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
    return cdist(xq, xp, metric='euclidean')


# ---- Code from Ahmed Fasih ---- https://gist.github.com/fasiha/fdb5cec2054e6f1c6ae35476045a0bbd
def nearest_positive_definite(A):
    """Find the nearest positive-definite matrix to input
    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].
    
    Parameters
    ----------
    A : array
        Matrix.
    
    Returns
    -------
    array
        Nearest positive-definite matrix to A.  
    
    Notes
    -----
    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = la.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(la.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(la.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3


def isPD(B):
    """Returns true when input is positive-definite, via Cholesky
    
    Parameters
    ----------
    B : array
        Matrix to test.
        
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

# ----


# from scipy.linalg import eigh
# EPSILON = np.finfo(float).eps
#
# def nearest_positive_definite(A):
#     """Find the nearest positive-definite matrix
#     """

#     B = (A.T + A)/2
#     w, v = eigh(B)
#     D = np.diag(np.maximum(w,EPSILON))
#     APD = v@D@v.T
#     return APD
