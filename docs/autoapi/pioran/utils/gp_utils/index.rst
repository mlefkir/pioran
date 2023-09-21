
gp_utils
========

.. py:module:: pioran.utils.gp_utils


Overview
--------


.. list-table:: Function
   :header-rows: 0
   :widths: auto
   :class: summarytable

   * - :py:obj:`EuclideanDistance <pioran.utils.gp_utils.EuclideanDistance>`\ (xq, xp)
     - Compute the Euclidean distance between two arrays.
   * - :py:obj:`nearest_positive_definite <pioran.utils.gp_utils.nearest_positive_definite>`\ (A)
     - Find the nearest positive-definite matrix to input.
   * - :py:obj:`decompose_triangular_matrix <pioran.utils.gp_utils.decompose_triangular_matrix>`\ (M)
     - Decompose a triangular matrix into a vector of unique values.
   * - :py:obj:`reconstruct_triangular_matrix <pioran.utils.gp_utils.reconstruct_triangular_matrix>`\ (unique, reverse_indexes, tril_indexes, n)
     - Recompose a triangular matrix from a vector of unique values.
   * - :py:obj:`isPD <pioran.utils.gp_utils.isPD>`\ (B)
     - Returns true when input is positive-definite, via Cholesky.


.. list-table:: Attributes
   :header-rows: 0
   :widths: auto
   :class: summarytable

   * - :py:obj:`valid_methods <pioran.utils.gp_utils.valid_methods>`
     - \-
   * - :py:obj:`tinygp_methods <pioran.utils.gp_utils.tinygp_methods>`
     - \-



Functions
---------
.. py:function:: EuclideanDistance(xq, xp)

   
   Compute the Euclidean distance between two arrays.

   .. math:: :label: euclidian_distance

       D(\boldsymbol{x_q},\boldsymbol{x_p}) = \sqrt{(\boldsymbol{x_q} - \boldsymbol{x_p}^{\mathrm{T}})^2}

   :Parameters:

       **xq** : (n, 1) :obj:`jax.Array`
           First array.

       **xp** : (m, 1) :obj:`jax.Array`
           Second array.

       **Returns**
           ..

       **-------**
           ..

       **(n, m) :obj:`jax.Array`**
           ..














   ..
       !! processed by numpydoc !!

.. py:function:: nearest_positive_definite(A)

   
   Find the nearest positive-definite matrix to input.

   Code from Ahmed Fasih - https://gist.github.com/fasiha/fdb5cec2054e6f1c6ae35476045a0bbd
   A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
   credits [2].

   :Parameters:

       **A** : (N, N) :obj:`jax.Array`
           Matrix to find the nearest positive-definite

   :Returns:

       (N, N) :obj:`jax.Array`
           Nearest positive-definite matrix to A.








   .. rubric:: Notes

   1. https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
   2. N.J. Higham, "Computing a nearest symmetric positive semidefinite" (1988): https://doi.org/10.1016/0024-3795(88)90223-6





   ..
       !! processed by numpydoc !!

.. py:function:: decompose_triangular_matrix(M)

   
   Decompose a triangular matrix into a vector of unique values.

   Decompose a triangular matrix into a vector of unique values and returns the
   indexes to reconstruct the original matrix.

   :Parameters:

       **M** : (n,n) :obj:`jax.Array`
           Triangular matrix of shape (n,n).

   :Returns:

       **unique** : :obj:`jax.Array`
           Vector of unique values.

       **reverse_indexes** : :obj:`jax.Array`
           Indexes to reconstruct the original matrix.

       **tril_indexes** : :obj:`jax.Array`
           Indexes of the lower triangular matrix.

       **n** : :obj:`int`
           Size of the original matrix.













   ..
       !! processed by numpydoc !!

.. py:function:: reconstruct_triangular_matrix(unique, reverse_indexes, tril_indexes, n)

   
   Recompose a triangular matrix from a vector of unique values.

   Recompose a triangular matrix from a vector of unique values and the indexes

   :Parameters:

       **unique** : :obj:`jax.Array`
           Vector of unique values.

       **reverse_indexes** : :obj:`jax.Array`
           Indexes to reconstruct the original matrix.

       **tril_indexes** : :obj:`jax.Array`
           Indexes of the lower triangular matrix.

       **n** : :obj:`int`
           Size of the original matrix.

   :Returns:

       :obj:`jax.Array`
           Triangular matrix of shape (n,n).




   :Raises:

       ValueError
           If the matrix is not triangular.









   ..
       !! processed by numpydoc !!

.. py:function:: isPD(B)

   
   Returns true when input is positive-definite, via Cholesky.


   :Parameters:

       **B** : (n,n) :obj:`jax.Array`
           Matrix to test.

   :Returns:

       :obj:`bool`
           `True` if B is positive-definite. `False` otherwise.













   ..
       !! processed by numpydoc !!


Attributes
----------
.. py:data:: valid_methods
   :value: ['FFT', 'NuFFT', 'SHO']

   

.. py:data:: tinygp_methods
   :value: ['SHO']

   



