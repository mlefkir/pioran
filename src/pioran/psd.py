import equinox as eqx
import jax.numpy as jnp

from .psd_base import PowerSpectralDensity
from .parameters import ParametersModel


class Lorentzian(PowerSpectralDensity):
    """Class for the Lorentzian power spectral density.
    
    
    Parameters
    ----------
    parameters_values : list
        list of the parameters values.
        in the order: [position, amplitude, halfwidth]
    kwargs : dict
        free_parameters: list of booleans
    """
    parameters: ParametersModel
    expression = 'lorentzian'
    
    def __init__(self, parameters_values, **kwargs):
        """Constructor of the Lorentzian class inheriting from PowerSpectralDensity.
        """
        assert len(parameters_values) == 3, 'The number of parameters for the lorentzian PSD must be 3'
        free_parameters = kwargs.get('free_parameters', [True, True,True])
        # initialise the parameters and check
        PowerSpectralDensity.__init__(self, param_values=parameters_values, param_names=["position",'amplitude', 'halfwidth'], free_parameters=free_parameters)
    
    def calculate(self,x) -> jnp.ndarray:
        # return self.parameters['amplitude'].value / ( ( 1 + ( ( x - self.parameters['position'].value ) / self.parameters['halfwidth'].value )**2 ) )/jnp.pi/self.parameters['halfwidth'].value
        # return 2 * self.parameters['amplitude'].value * self.parameters['halfwidth'].value  /  ( self.parameters['halfwidth'].value**2 + 4 * jnp.pi**2 * ( x - self.parameters['position'].value )**2 )
        return self.parameters['amplitude'].value  /  ( self.parameters['halfwidth'].value**2 + 4 * jnp.pi**2 * ( x - self.parameters['position'].value )**2 )

class Gaussian(PowerSpectralDensity):
    r"""
    
            K(\tau) = A \times \exp{\left( -\dfrac{\tau^2}{2\sigma}\right) } 


    Parameters
    ----------
    PowerSpectralDensity : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    componentname = 'gaussian'
    ID = 1
    n_parameters = 3 
    
    def __init__(self, parameters_values, **kwargs):
        """
        """
        assert len(parameters_values) == self.n_parameters, 'The number of parameters for this  covariance function must be 3'
        free_parameters = kwargs.get('free_parameters', [True, True,True])
        # initialise the parameters and check
        PowerSpectralDensity.__init__(self, parameters_values, names=["position",'amplitude', 'sigma'], boundaries=[[-jnp.inf, jnp.inf], [0, jnp.inf],[0,jnp.inf]], free_parameters=free_parameters)
    
    def calculate(self,x) -> jnp.ndarray:
        return self.parameters['amplitude'].value / (jnp.sqrt( 2*jnp.pi ) * self.parameters['sigma'].value ) * jnp.exp( -0.5 * (x - self.parameters['position'].value )**2 / self.parameters['sigma'].value**2 )


class Rectangular(PowerSpectralDensity):
    """
    
    Weaknesses:
    Will not be useful for performance because the function multiplied by the rectangular function will
    still be evaluated at all points. Need to find a way to make this more efficient (e.g. using ast to make the parsing 
    to compile more efficient)
    

    Parameters
    ----------
    PowerSpectralDensity : _type_
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
        PowerSpectralDensity.__init__(self, parameters_values, names=['min','max', 'amplitude'], boundaries=[[-jnp.inf, jnp.inf], [0, jnp.inf],[0,jnp.inf]], free_parameters=free_parameters)
    
    def calculate(self,x):
        return jnp.where((x>=self.parameters['min'].value)&(x<self.parameters['max'].value),self.parameters['amplitude'].value,0)

class PowerLawLim(PowerSpectralDensity):
    componentname = 'powerlaw'
    ID = 1
    n_parameters = 5
    
    def __init__(self, parameters_values, **kwargs):
        """
        """
        assert len(parameters_values) == self.n_parameters, f'The number of parameters for {self.__classname__()} must be {self.n_parameters}, not {len(parameters_values)}'
        free_parameters = kwargs.get('free_parameters', [True,True,True, True,True])
        # initialise the parameters and check
        PowerSpectralDensity.__init__(self, parameters_values, names=['min','max','freq','amplitude', 'index'], boundaries=[[0, jnp.inf], [0, jnp.inf],[0, jnp.inf], [0, jnp.inf],[0,jnp.inf]], free_parameters=free_parameters)
    
    def calculate(self,x):
        return jnp.where((x>=self.parameters['min'].value)&(x<self.parameters['max'].value),self.parameters['amplitude'].value * jnp.power( x / self.parameters['freq'].value , -self.parameters['index'].value ), 0)
    
class Scalar(PowerSpectralDensity):
    componentname = 'scalar'
    ID = 1
    n_parameters = 1
    def __init__(self, parameters_values, **kwargs):
        """
        """
        assert len(parameters_values) == self.n_parameters, f'The number of parameters for {self.__classname__()} must be {self.n_parameters}, not {len(parameters_values)}'
        free_parameters = kwargs.get('free_parameters', [True])
        # initialise the parameters and check
        PowerSpectralDensity.__init__(self, parameters_values, names=['scalar'], boundaries=[[-jnp.inf, jnp.inf]], free_parameters=free_parameters)
    def calculate(self,x):
        return self.parameters['scalar'].value

class PowerLaw(PowerSpectralDensity):
    componentname = 'powerlaw'
    ID = 1
    n_parameters = 3
    
    def __init__(self, parameters_values, **kwargs):
        """
        """
        assert len(parameters_values) == self.n_parameters, f'The number of parameters for {self.__classname__()} must be {self.n_parameters}, not {len(parameters_values)}'
        free_parameters = kwargs.get('free_parameters', [True, True,True])
        # initialise the parameters and check
        PowerSpectralDensity.__init__(self, parameters_values, names=['freq','amplitude', 'index'], boundaries=[[0, jnp.inf], [0, jnp.inf],[0,jnp.inf]], free_parameters=free_parameters)
    
    def calculate(self,x):
        return self.parameters['amplitude'].value * jnp.power( x / self.parameters['freq'].value , -self.parameters['index'].value )
    
class MultipleBendingPowerLaw(PowerSpectralDensity):
    """Power spectrum model for the multiple bending power law
    
    P = A * (f/f0)^(-n) * Product (from i = 1 to N) (1 + (f/f_{B_i})^(alpha_{i+1}-alpha_i)))^{-1}

    """
    
    
    componentname = 'multiplebendingpowerlaw'
    ID = 1
    
    
    def __init__(self, parameters_values, **kwargs):
        """
        """
        
        assert len(parameters_values) %2 == 1 and len(parameters_values)>=4 , f'The number of parameters for {self.__classname__()} must be greater than 4 and even, not {len(parameters_values)}'
        self.N = len(parameters_values)//2-1
        free_parameters = kwargs.get('free_parameters', [True]*len(parameters_values))
        # initialise the parameters and check
        self.n_parameters = len(parameters_values)
        names=['amplitude', 'freq_1', 'index_1']
        
        [(names.append(f'freq_{i+1}'),names.append(f'index_{i+1}')) for i in range(1,1+self.N)]
        
        PowerSpectralDensity.__init__(self, parameters_values, names=names, boundaries=[[0,jnp.inf]]*self.n_parameters, free_parameters=free_parameters)
                                    
    def calculate(self,x):
        P = self.parameters[f'amplitude'].value/ jnp.power( x / self.parameters[f'freq_1'].value , self.parameters[f'index_1'].value )
        for i in range(1,1+self.N):
            P /=   (1 + jnp.power( x / self.parameters[f'freq_{i+1}'].value , self.parameters[f'index_{i+1}'].value-self.parameters[f'index_{i}'].value ) )
        return P