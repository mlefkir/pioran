"""All the parameters of either :class:`~pioran.acvf_base.CovarianceFunction` or :class:`~pioran.psd_base.PowerSpectralDensity`
are stored in a :class:`~pioran.parameters.ParametersModel` object which contains several instances of :class:`~pioran.parameters.Parameter`. 

The base class for all parameters. By construction as a JAX Pytree_, some of the attributes are frozen and cannot be changed during runtime.
The attributes that can be changed are the ``name`` and ``value`` of the parameter. The parameter can be considered as *free* or *fixed* depending on the value of the boolean attribute ``free``.
A parameter can be referred to by its ``name`` or by index ``ID``. The ``ID`` is the index of the parameter in the list of parameters of the :class:`~pioran.parameters.ParametersModel` object.

The attribute ``component`` is used to refer to parameters in individual components together when combining several model components.
The attribute ``hyperparameter`` is used to refer to parameters that are not directly related to the model via covariance functions or power spectral densities.

.. _Pytree: https://jax.readthedocs.io/en/latest/pytrees.html


On top of the base class is built a :class:`~pioran.parameters.ParametersModel` object. This object inherits from :class:`equinox.Module`, which means
the attributes of the :class:`~pioran.parameters.ParametersModel` object are immutable and cannot be changed during runtime.

The values of the free parameters can be changed during runtime using the method :meth:`~pioran.parameters.ParametersModel.set_free_values`. 
The names of the parameters can be changed using the method :meth:`~pioran.parameters.ParametersModel.set_names`.
The values, names, IDs and free status of the parameters can be accessed using attributes. 

The :class:`~pioran.parameters.Parameter` stored in :class:`~pioran.parameters.Parameter` can be accessed by the name of the parameter or by index with the ``[]`` operator. 
If there are several parameters with the same name, the first one is returned.

It is possible to add new parameters to the :class:`~pioran.parameters.ParametersModel` object using the method :meth:`~pioran.parameters.ParametersModel.append`.
"""
from .parameters import Parameter, ParametersModel

__all__ = ["Parameter", "ParametersModel"]