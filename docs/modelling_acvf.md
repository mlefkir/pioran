<!-- # Modelling with an Autocovariance function

In this package, we can model the underlying stochastic process which might generate the time series using either a model for the autocovariance function or the power spectral density function.


The autocovariance function of the time series is represented by the object {class}`~pioran.acvf_base.CovarianceFunction`. More details about this class and the implemented models can be found [here](apidocs/ACVF). The following code shows how to create a new model component for the autocovariance function. 

## Simple models



## Combining autocovariance functions

In this section, we show how to combine two autocovariance functions via arithmetic operations. This is useful when we want to model the autocovariance function of a time series as the sum of two autocovariance functions. For instance, we might want to model the autocovariance function of a time series as the sum of a long-term autocovariance function and a short-term autocovariance function.


## Writing a new model for the autocovariance function

### Creating the class and the constructor
We write a new class ``MyAutocovariance``, which inherits from `CovarianceFunction`. It is important to specify the attributes ``parameters`` and ``expression`` at the class level as `CovarianceFunction` inherits from {class}`~equinox.Module`. ``parameters`` is an object of the class {class}`~pioran.parameters.ParametersModel` which is a container for the parameters of the model. The constructor ``__init__`` must be defined as in the example and the names of the parameters are given in the ``param_names`` list. 


### The ``calculate`` method
The ``calculate`` method must be defined and it must return the autocovariance function evaluated at the time ``t``. When writting the expression of the autocovariance function, the values of parameters of the model can be accessed using the attribute ``self.parameters['name'].value`` where ``name`` is the name of the parameter.

This method is then called by the method {meth}`~pioran.acvf_base.CovarianceFunction.get_cov_matrix` to compute for instance the likelihood or the posterior predictive distribution of a Gaussian process.

```python
class MyAutocovariance(CovarianceFunction):
    parameters: ParametersModel
    expression = 'name of the model'

    def __init__(self, param_values, **kwargs):
        """Constructor of the covariance function inherited 
        from the CovarianceFunction class.
        """
        assert len(param_values) == 2, 'The number of parameters must be 2'
        free_parameters = kwargs.get('free_parameters', [True, True])
        CovarianceFunction.__init__(self, param_values=param_values, 
        param_names=['variance', 'length'], free_parameters=free_parameters)
    
    def calculate(self,t) -> jnp.array:
        """Returns the autocovariance function evaluated at t.
        """
        return  self.parameters['variance'].value *jnp.exp(- jnp.abs(t) * self.parameters['length'].value)
``` -->
