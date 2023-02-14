
import equinox as eqx
import jax.numpy as jnp
from .parameters import ParametersModel

class PowerSpectralDensity(eqx.Module):
    """ Master class for the power density function functions.
    
    """
    parameters: ParametersModel
    expression: str
    
    def __init__(self, param_values,param_names, free_parameters):

        if isinstance(param_values, ParametersModel):
            self.parameters = param_values
        elif isinstance(param_values, list) or isinstance(param_values, jnp.ndarray):
            self.parameters = ParametersModel( param_names=param_names, param_values=param_values, free_parameters=free_parameters)
        else:
            raise TypeError(
                "The parameters of the power spectral density must be a list of floats or jnp.ndarray or a ParametersModel object.")

    def __str__(self)->str:
        """String representation of the power spectral density.
        
        Returns
        -------
        str
            String representation of the power spectral density.
        """
    
        s = f"Power spectrum: {self.expression}\n"
        s += f"Number of parameters: {len(self.parameters.values)}\n"
        s += self.parameters.__str__()
        return s
    
    
    
    def __add__(self, other)->"SumPowerSpectralDensity":
        """Overload of the + operator for the power spectral densities.

        Parameters
        ----------
        other : :obj:`PowerSpectralDensity`
            power spectral density to add.

        Returns
        -------
        :obj:`SumPowerSpectralDensity`
            Sum of the two power spectral densities.
        """
        other.parameters.increment_IDs(len(self.parameters.values))
        other.parameters.increment_component(max(self.parameters.components))
        return SumPowerSpectralDensity(self, other)
    
    def __mul__(self, other)->"ProductPowerSpectralDensity":
        """Overload of the * operator for the power spectral densities.
        
        Parameters
        ----------
        other : :obj:`PowerSpectralDensity`
            power spectral density to multiply.
        
        Returns
        -------
        :obj:`ProductPowerSpectralDensity`
            Product of the two power spectral densities.
        """
        
        other.parameters.increment_IDs(len(self.parameters.values))
        other.parameters.increment_component(max(self.parameters.components))
        return ProductPowerSpectralDensity(self, other)
    
class ProductPowerSpectralDensity(PowerSpectralDensity):
    """Class for the product of two power spectral densities.

    Parameters
    ----------
    psd1 : :obj:`PowerSpectralDensity`
        First power spectral density.
    psd2 : :obj:`PowerSpectralDensity`
        Second power spectral density.

    Attributes
    ----------
    psd1 : :obj:`PowerSpectralDensity`
        First power spectral density.
    psd2 : :obj:`PowerSpectralDensity`
        Second power spectral density.
    parameters : :obj:`ParametersModel`
        Parameters of the power spectral density.
    expression : str
        Expression of the total power spectral density.

    """
    psd1: PowerSpectralDensity
    psd2: PowerSpectralDensity
    parameters: ParametersModel
    expression: str
    
    def __init__(self, psd1, psd2):
        """Constructor of the SumPowerSpectralDensity class."""
        self.psd1 = psd1
        self.psd2 = psd2
        if isinstance(psd1, SumPowerSpectralDensity) and isinstance(psd2, SumPowerSpectralDensity):
            self.expression = f'({psd1.expression}) * ({psd2.expression})'
        elif isinstance(psd1, SumPowerSpectralDensity):
            self.expression = f'({psd1.expression}) * {psd2.expression}'
        elif isinstance(psd2, SumPowerSpectralDensity):
            self.expression = f'{psd1.expression} * ({psd2.expression})'
        else:
            self.expression = f'{psd1.expression} * {psd2.expression}'     
        
        
        self.parameters = ParametersModel(param_names=psd1.parameters.names + psd2.parameters.names,
                                          param_values=psd1.parameters.values + psd2.parameters.values,
                                          free_parameters=psd1.parameters.free_parameters + psd2.parameters.free_parameters,
                                          _pars=psd1.parameters._pars + psd2.parameters._pars)
    @eqx.filter_jit
    def calculate(self, x):
        """Compute the power spectral density at the points x.
        
        It is the product of the two power spectral densities.
        
        Parameters
        ----------
        x : array 
            Points where the power spectral density is computed.
        
        Returns
        -------
        Product of the two power spectral densitys at the points x.
        
        """
        return self.psd1.calculate(x) * self.psd2.calculate(x)
    
class SumPowerSpectralDensity(PowerSpectralDensity):
    """Class for the sum of two power spectral densitys.

    Parameters
    ----------
    psd1 : :obj:`PowerSpectralDensity`
        First power spectral density.
    psd2 : :obj:`PowerSpectralDensity`
        Second power spectral density.

    Attributes
    ----------
    psd1 : :obj:`PowerSpectralDensity`
        First power spectral density.
    psd2 : :obj:`PowerSpectralDensity`
        Second power spectral density.
    parameters : :obj:`ParametersModel`
        Parameters of the power spectral density.
    expression : str
        Expression of the total power spectral density.

    """
    psd1: PowerSpectralDensity
    psd2: PowerSpectralDensity
    parameters: ParametersModel
    expression: str
    
    def __init__(self, psd1, psd2):
        """Constructor of the SumPowerSpectralDensity class."""
        self.psd1 = psd1
        self.psd2 = psd2
        self.expression = f'{psd1.expression} + {psd2.expression}'
        
        self.parameters = ParametersModel(param_names=psd1.parameters.names + psd2.parameters.names,
                                          param_values=psd1.parameters.values + psd2.parameters.values,
                                          free_parameters=psd1.parameters.free_parameters + psd2.parameters.free_parameters,
                                          _pars=psd1.parameters._pars + psd2.parameters._pars)
    @eqx.filter_jit
    def calculate(self, x):
        """Compute the power spectrum at the points x.
        
        It is the sum of the two power spectra.
        
        Parameters
        ----------
        x : array 
            Points where the power spectrum is computed.
        
        Returns
        -------
        Sum of the two power spectra at the points x.
        
        """
        return self.psd1.calculate(x) + self.psd2.calculate(x)
