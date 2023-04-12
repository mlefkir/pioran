Power spectral density functions
================================
 
 .. currentmodule:: pioran.psd_base

Classes for manipulating and creating power spectral density functions (PSDs).


Base class
----------

This is the base class for all PSD functions. It is not meant to be used directly, but rather as a base class for other PSD functions. 
The sum and product of PSD functions are implemented with the ``+`` and ``*`` operators, respectively.

.. autosummary::
    :toctree: summary
    :recursive:
    :nosignatures:

    PowerSpectralDensity


Implemented PSD functions
-------------------------

These are the PSD functions that are currently implemented. They all inherit from the base class. 

..  currentmodule:: pioran.psd

.. autosummary::
    :toctree: summary
    :nosignatures:  
    :recursive:

    Lorentzian
    Gaussian
    Matern32PSD
    MultipleBendingPowerLaw



Power spectrum to autocovariance
--------------------------------

It is possible to convert a power spectrum to an autocovariance function. This is done using the Discrete Fourier Transform and interpolating the result 
to the desired time lags.

..  currentmodule:: pioran.psdtoacv

.. autosummary::
    :toctree: summary
    :recursive:
    :nosignatures:

    PSDToACV

