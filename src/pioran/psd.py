import jax
import jax.numpy as jnp

from .parameters import ParametersModel
from .psd_base import PowerSpectralDensity

class Lorentzian(PowerSpectralDensity):
    """Class for the Lorentzian power spectral density.
    
    .. math:: :label: lorentzianpsd 
    
       \mathcal{P}(f) = \dfrac{A}{\gamma^2 +4\pi^2 (f-f_0)^2}.

    with the amplitude :math:`A\ge 0`, the position :math:`f_0\ge 0` and the halfwidth :math:`\gamma>0`.
    
    The parameters are stored in the `parameters` attribute which is a :class:`~pioran.parameters.ParametersModel` object. 
    The values of the parameters can be accessed using the `parameters` attribute via three keys: '`position`', '`amplitude`' and '`halfwidth`'.
    
    The power spectral density function is evaluated on an array of frequencies :math:`f` using the `calculate` method.
    
    
    Parameters
    ----------
    param_values : :obj:`list of float`
        Values of the parameters of the power spectral density function. [position, amplitude, halfwidth]
    free_parameters: :obj:`list of bool`, optional
        List of bool to indicate if the parameters are free or not. Default is `[True, True,True]`.
            
    Attributes
    ----------
    parameters : :class:`~pioran.parameters.ParametersModel`
        Parameters of the power spectral density function.
        
    Methods
    -------
    calculate(t)
        Computes the power spectral density function on an array of frequencies :math:`f`.
    """
    parameters: ParametersModel
    expression = 'lorentzian'
    analytical = True
    
    def __init__(self, parameters_values, free_parameters = [True, True,True]):
        assert len(parameters_values) == 3, 'The number of parameters for the lorentzian PSD must be 3'
        # initialise the parameters and check
        PowerSpectralDensity.__init__(self, param_values=parameters_values, param_names=["position",'amplitude', 'halfwidth'], free_parameters=free_parameters)
    
    def calculate(self,f) -> jax.Array:
        r"""Computes the Lorentzian power spectral density function on an array of frequencies :math:`f`.
        
        The expression is given by Equation :math:numref:`lorentzianpsd`.
        with the variance :math:`A\ge 0`, the position :math:`f_0\ge 0` and the halfwidth :math:`\gamma>0`.

        Parameters
        ----------
        f : :obj:`jax.Array`
            Array of frequencies.

        Returns
        -------
        :obj:`jax.Array`
            Power spectral density function evaluated on the array of frequencies.
        """
        return self.parameters['amplitude'].value  /  ( self.parameters['halfwidth'].value**2 + 4 * jnp.pi**2 * ( f - self.parameters['position'].value )**2 )

class Gaussian(PowerSpectralDensity):
    r""" Class for the Gaussian power spectral density.

    .. math:: :label: gaussianpsd 
    
       \mathcal{P}(f) = \dfrac{A}{\sqrt{2\pi}\sigma} \exp\left(-\dfrac{\left(f-f_0\right)^2}{2\sigma^2} \right).

    with the amplitude :math:`A\ge 0`, the position :math:`f_0\ge 0` and the standard-deviation '`sigma`' :math:`\sigma>0`.
    
    The parameters are stored in the `parameters` attribute which is a :class:`~pioran.parameters.ParametersModel` object. 
    The values of the parameters can be accessed using the `parameters` attribute via three keys: '`position`', '`amplitude`' and '`sigma`'
    
    The power spectral density function is evaluated on an array of frequencies :math:`f` using the `calculate` method.
    
    
    Parameters
    ----------
    param_values : :obj:`list of float`
        Values of the parameters of the power spectral density function.
    free_parameters: :obj:`list of bool`, optional
        List of bool to indicate if the parameters are free or not. Default is `[True, True,True]`.
            
    Attributes
    ----------
    parameters : :class:`~pioran.parameters.ParametersModel`
        Parameters of the power spectral density function.
    expression : :obj:`str`
        Expression of the power spectral density function.
    analytical : :obj:`bool`
        If True, the power spectral density function is analytical, otherwise it is not.
        
    Methods
    -------
    calculate(t)
        Computes the power spectral density function on an array of frequencies :math:`f`.
    """
    expression = 'gaussian'
    parameters: ParametersModel
    analytical = True
    
    def __init__(self, parameters_values, free_parameters=[True, True,True]):
        assert len(parameters_values) == 3, f'The number of parameters for the power spectral density function "{self.expression}" must be 3'
        # initialise the parameters and check
        PowerSpectralDensity.__init__(self, param_values=parameters_values, param_names=["position",'amplitude', 'sigma'], free_parameters=free_parameters)
    
    def calculate(self,f) -> jax.Array:
        r"""Computes the Gaussian power spectral density function on an array of frequencies :math:`f`.
        
        The expression is given by Equation :math:numref:`gaussianpsd` 
        with the variance :math:`A\ge 0`, the position :math:`f_0\ge 0` and the standard-deviation :math:`\sigma>0`.

        Parameters
        ----------
        f : :obj:`jax.Array`
            Array of frequencies.

        Returns
        -------
        :obj:`jax.Array`
            Power spectral density function evaluated on the array of frequencies.
        """
        return self.parameters['amplitude'].value / (jnp.sqrt( 2*jnp.pi ) * self.parameters['sigma'].value ) * jnp.exp( -0.5 * (f - self.parameters['position'].value )**2 / self.parameters['sigma'].value**2 )

    
class OneBendingPowerLawNorm(PowerSpectralDensity):
    expression = 'onebendingpowerlaw_norm'
    parameters: ParametersModel    

    def __init__(self, parameters_values, free_parameters = [True, True,True,True]):     
        assert len(parameters_values) == 4 , f'The number of parameters for {self.__classname__()} must be 4, not {len(parameters_values)}'
        assert len(free_parameters) == 4, f'The number of free parameters for {self.__classname__()} must be 4, not {len(free_parameters)}'
        # initialise the parameters and check
        names=['norm','index_1', 'freq_1', 'index_2']
                
        PowerSpectralDensity.__init__(self, param_values=parameters_values, param_names=names, free_parameters=free_parameters)
                                    
    def calculate(self,f):
        r"""Computes the Multiple bending power-law model on an array of frequencies :math:`f`.
        
        The expression is given by Equation :math:numref:`multiplebendplpsd`
        with the variance :math:`A\ge 0` and the scale :math:`\gamma>0`.

        Parameters
        ----------
        f : :obj:`jax.Array`
            Array of frequencies.

        Returns
        -------
        :obj:`jax.Array`
            Power spectral density function evaluated on the array of frequencies.
        """
        
        index_1,f_1,index_2,norm = self.parameters['index_1'].value, self.parameters['freq_1'].value, self.parameters['index_2'].value, self.parameters['norm'].value
        P = jnp.power( f/f_1 , index_1 ) * jnp.power( 1 + jnp.power ( f / f_1 , index_1-index_2 ),-1 )
        return P*norm
        
class OneBendPowerLaw(PowerSpectralDensity):
    """Class for the bending power-law power spectral density with one bend.
    
    
    .. math:: :label: onebendpowerlawpsd

         \mathcal{P}(f) = \dfrac{A}{\left(1+\left(\dfrac{f}{f_1}\right)^{\alpha_1-\alpha_2}\right)} \left(\dfrac{f}{f_1}\right)^{\alpha_1}.
             
    
    """
    expression = 'onebendpowerlaw'
    parameters: ParametersModel    
    
    def __init__(self, parameters_values, free_parameters = [True, True,True]):     
        assert len(parameters_values) == 3 , f'The number of parameters for {self.__classname__()} must be 3, not {len(parameters_values)}'
        assert len(free_parameters) == 3, f'The number of free parameters for {self.__classname__()} must be 3, not {len(free_parameters)}'

        # initialise the parameters and check
        names=['index_1', 'freq_1', 'index_2']
                
        PowerSpectralDensity.__init__(self, param_values=parameters_values, param_names=names, free_parameters=free_parameters)
                                    
    def calculate(self,f):
        r"""Computes the Multiple bending power-law model on an array of frequencies :math:`f`.
        
        The expression is given by Equation :math:numref:`multiplebendplpsd`
        with the variance :math:`A\ge 0` and the scale :math:`\gamma>0`.

        Parameters
        ----------
        f : :obj:`jax.Array`
            Array of frequencies.

        Returns
        -------
        :obj:`jax.Array`
            Power spectral density function evaluated on the array of frequencies.
        """
        
        index_1,f_1,index_2 = self.parameters['index_1'].value, self.parameters['freq_1'].value, self.parameters['index_2'].value
        P = jnp.power( f/f_1 , index_1 ) * jnp.power( 1 + jnp.power ( f / f_1 , index_1-index_2 ),-1 )
        return P