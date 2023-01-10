import jax.numpy as jnp
from .psd_base import PowerSpectralDensityComponent

class Lorentzian(PowerSpectralDensityComponent):
    componentname = 'lorentzian'
    ID = 1
    n_parameters = 3 
    
    def __init__(self, parameters_values, **kwargs):
        """
        """
        assert len(parameters_values) == self.n_parameters, 'The number of parameters for this  covariance function must be 3'
        free_parameters = kwargs.get('free_parameters', [True, True,True])
        # initialise the parameters and check
        PowerSpectralDensityComponent.__init__(self, parameters_values, names=["position",'amplitude', 'halfwidth'], boundaries=[[-jnp.inf, jnp.inf], [0, jnp.inf],[0,jnp.inf]], free_parameters=free_parameters)
    
    def fun(self,x) -> jnp.ndarray:
        return self.parameters['amplitude'].value / ( ( 1 + ( ( x - self.parameters['position'].value ) / self.parameters['halfwidth'].value )**2 ) )

class Gaussian(PowerSpectralDensityComponent):
    componentname = 'gaussian'
    ID = 1
    n_parameters = 3 
    
    def __init__(self, parameters_values, **kwargs):
        """
        """
        assert len(parameters_values) == self.n_parameters, 'The number of parameters for this  covariance function must be 3'
        free_parameters = kwargs.get('free_parameters', [True, True,True])
        # initialise the parameters and check
        PowerSpectralDensityComponent.__init__(self, parameters_values, names=["position",'amplitude', 'sigma'], boundaries=[[-jnp.inf, jnp.inf], [0, jnp.inf],[0,jnp.inf]], free_parameters=free_parameters)
    
    def fun(self,x) -> jnp.ndarray:
        return self.parameters['amplitude'].value / (jnp.sqrt( 2*jnp.pi ) * self.parameters['sigma'].value ) * jnp.exp( 0.5 * (x - self.parameters['position'].value )**2 / self.parameters['sigma'].value**2 )




class Rectangular(PowerSpectralDensityComponent):
    """
    
    Weaknesses:
    Will not be useful for performance because the function multiplied by the rectangular function will
    still be evaluated at all points. Need to find a way to make this more efficient (e.g. using ast to make the parsing 
    to compile more efficient)
    

    Parameters
    ----------
    PowerSpectralDensityComponent : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    
    componentname = 'rectangular'
    ID = 1
    n_parameters = 3
    
    def __init__(self, parameters_values, **kwargs):
        """
        """
        assert len(parameters_values) == self.n_parameters, f'The number of parameters for {self.__classname__()} must be {self.n_parameters}, not {len(parameters_values)}'
        free_parameters = kwargs.get('free_parameters', [True, True,True])
        # initialise the parameters and check
        PowerSpectralDensityComponent.__init__(self, parameters_values, names=['min','max', 'amplitude'], boundaries=[[-jnp.inf, jnp.inf], [0, jnp.inf],[0,jnp.inf]], free_parameters=free_parameters)
    
    def fun(self,x):
        return jnp.where((x>=self.parameters['min'].value)&(x<self.parameters['max'].value),self.parameters['amplitude'].value,0)

class PowerLawLim(PowerSpectralDensityComponent):
    componentname = 'powerlaw'
    ID = 1
    n_parameters = 5
    
    def __init__(self, parameters_values, **kwargs):
        """
        """
        assert len(parameters_values) == self.n_parameters, f'The number of parameters for {self.__classname__()} must be {self.n_parameters}, not {len(parameters_values)}'
        free_parameters = kwargs.get('free_parameters', [True,True,True, True,True])
        # initialise the parameters and check
        PowerSpectralDensityComponent.__init__(self, parameters_values, names=['min','max','freq','amplitude', 'index'], boundaries=[[0, jnp.inf], [0, jnp.inf],[0, jnp.inf], [0, jnp.inf],[0,jnp.inf]], free_parameters=free_parameters)
    
    def fun(self,x):
        return jnp.where((x>=self.parameters['min'].value)&(x<self.parameters['max'].value),self.parameters['amplitude'].value * jnp.power( x / self.parameters['freq'].value , -self.parameters['index'].value ), 0)
    
class Scalar(PowerSpectralDensityComponent):
    componentname = 'scalar'
    ID = 1
    n_parameters = 1
    def __init__(self, parameters_values, **kwargs):
        """
        """
        assert len(parameters_values) == self.n_parameters, f'The number of parameters for {self.__classname__()} must be {self.n_parameters}, not {len(parameters_values)}'
        free_parameters = kwargs.get('free_parameters', [True])
        # initialise the parameters and check
        PowerSpectralDensityComponent.__init__(self, parameters_values, names=['scalar'], boundaries=[[-jnp.inf, jnp.inf]], free_parameters=free_parameters)
    def fun(self,x):
        return self.parameters['scalar'].value

class PowerLaw(PowerSpectralDensityComponent):
    componentname = 'powerlaw'
    ID = 1
    n_parameters = 3
    
    def __init__(self, parameters_values, **kwargs):
        """
        """
        assert len(parameters_values) == self.n_parameters, f'The number of parameters for {self.__classname__()} must be {self.n_parameters}, not {len(parameters_values)}'
        free_parameters = kwargs.get('free_parameters', [True, True,True])
        # initialise the parameters and check
        PowerSpectralDensityComponent.__init__(self, parameters_values, names=['freq','amplitude', 'index'], boundaries=[[0, jnp.inf], [0, jnp.inf],[0,jnp.inf]], free_parameters=free_parameters)
    
    def fun(self,x):
        return self.parameters['amplitude'].value * jnp.power( x / self.parameters['freq'].value , -self.parameters['index'].value )
    
