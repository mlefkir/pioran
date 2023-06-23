Covariance functions
====================

Classes for manipulating and creating covariance functions.
 
 .. currentmodule:: pioran.acvf_base


Base class
----------

This is the base class for all covariance functions. It is not meant to be used directly, but rather as a base class for other covariance functions. 
The sum and product of covariance functions are implemented with the ``+`` and ``*`` operators, respectively.

.. autosummary::
    :toctree: summary
    :recursive:
    :nosignatures:

    CovarianceFunction


Implemented covariance functions
--------------------------------

These are the covariance functions that are currently implemented, they inherit from the base class.

..  currentmodule:: pioran.acvf

.. autosummary::
    :toctree: summary
    :nosignatures:  
    :recursive:

    Exponential
    SquaredExponential
    Matern32
    Matern52
    RationalQuadratic
    CARMA_covariance