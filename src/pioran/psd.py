import jax.numpy as jnp

from .parameters import ParametersModel
from .psd_base import PowerSpectralDensity
import scipy.special as sp

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
        Values of the parameters of the power spectral density function.
    **kwargs : :obj:`dict`        
        free_parameters: :obj:`list of bool`
            List of bool to indicate if the parameters are free or not.
            
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
    
    def __init__(self, parameters_values, **kwargs):
        assert len(parameters_values) == 3, 'The number of parameters for the lorentzian PSD must be 3'
        free_parameters = kwargs.get('free_parameters', [True, True,True])
        # initialise the parameters and check
        PowerSpectralDensity.__init__(self, param_values=parameters_values, param_names=["position",'amplitude', 'halfwidth'], free_parameters=free_parameters)
    
    def calculate(self,f) -> jnp.ndarray:
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
        # return self.parameters['amplitude'].value / ( ( 1 + ( ( x - self.parameters['position'].value ) / self.parameters['halfwidth'].value )**2 ) )/jnp.pi/self.parameters['halfwidth'].value
        # return 2 * self.parameters['amplitude'].value * self.parameters['halfwidth'].value  /  ( self.parameters['halfwidth'].value**2 + 4 * jnp.pi**2 * ( x - self.parameters['position'].value )**2 )
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
    **kwargs : :obj:`dict`        
        free_parameters: :obj:`list of bool`
            List of bool to indicate if the parameters are free or not.
            
    Attributes
    ----------
    parameters : :class:`~pioran.parameters.ParametersModel`
        Parameters of the power spectral density function.
        
    Methods
    -------
    calculate(t)
        Computes the power spectral density function on an array of frequencies :math:`f`.
    """
    expression = 'gaussian'
    parameters: ParametersModel
    
    def __init__(self, parameters_values, **kwargs):
        assert len(parameters_values) == 3, f'The number of parameters for the power spectral density function "{self.expression}" must be 3'
        free_parameters = kwargs.get('free_parameters', [True, True,True])
        # initialise the parameters and check
        PowerSpectralDensity.__init__(self, param_values=parameters_values, param_names=["position",'amplitude', 'sigma'], free_parameters=free_parameters)
    
    def calculate(self,f) -> jnp.ndarray:
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

class Matern32PSD(PowerSpectralDensity):
    """Class for the power spectral density of the Matern 3/2 covariance function.
    
    .. math:: :label: matern32psd 
    
       \mathcal{P}(f) = \dfrac{A}{\gamma^3}\dfrac{12\sqrt{3}}{{(3/\gamma^2 +4\pi^2 f^2)}^2}.

    with the amplitude :math:`A\ge 0` and the scale :math:`\gamma>0`.
    
    The parameters are stored in the `parameters` attribute which is a :class:`~pioran.parameters.ParametersModel` object. 
    The values of the parameters can be accessed using the `parameters` attribute via three keys: '`position`' and '`scale`'
    
    The power spectral density function is evaluated on an array of frequencies :math:`f` using the `calculate` method.
    
    
    Parameters
    ----------
    param_values : :obj:`list of float`
        Values of the parameters of the power spectral density function.
    **kwargs : :obj:`dict`        
        free_parameters: :obj:`list of bool`
            List of bool to indicate if the parameters are free or not.
            
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
    expression = 'matern32psd'
    
    def __init__(self, parameters_values, **kwargs):
        assert len(parameters_values) == 2, 'The number of parameters for the Matern3/2 PSD must be 2'
        free_parameters = kwargs.get('free_parameters', [True, True,True])
        # initialise the parameters and check
        PowerSpectralDensity.__init__(self, param_values=parameters_values, param_names=["amplitude",'scale'], free_parameters=free_parameters)
    
    def calculate(self,f) -> jnp.ndarray:
        r"""Computes the power spectral density of the Matern 3/2 covariance function on an array of frequencies :math:`f`.
        
        The expression is given by Equation :math:numref:`matern32psd`
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
        return self.parameters['amplitude'].value  * 12 * jnp.sqrt(3) / self.parameters['scale'].value**3 /  ( 3 / self.parameters['scale'].value**2 + 4 * jnp.pi**2 * f**2 )**2

class MultipleBendingPowerLaw(PowerSpectralDensity):
    r""" Class for the Multiple bending power-law power spectral density.

    .. math:: :label: multiplebendplpsd     
    
       \mathcal{P}(f) =  \dfrac{A \left({f/}{f_0}\right)^{-\alpha_0}}{\displaystyle\prod_{i=1}^{N} \left(1+\left(\dfrac{f}{f_{b_i}}\right)^{\alpha_{i+1}-\alpha_{i}}\right)}
    
    with the amplitude :math:`A\ge 0`, the position :math:`f_0\ge 0` and the standard-deviation '`sigma`' :math:`\sigma>0`.
    
    The parameters are stored in the `parameters` attribute which is a :class:`~pioran.parameters.ParametersModel` object. 
    The values of the parameters can be accessed using the `parameters` attribute via three keys: '`position`', '`amplitude`' and '`sigma`'
    
    The power spectral density function is evaluated on an array of frequencies :math:`f` using the `calculate` method.
    
    
    Parameters
    ----------
    param_values : :obj:`list of float`
        Values of the parameters of the power spectral density function.
    **kwargs : :obj:`dict`        
        free_parameters: :obj:`list of bool`
            List of bool to indicate if the parameters are free or not.
            
    Attributes
    ----------
    parameters : :class:`~pioran.parameters.ParametersModel`
        Parameters of the power spectral density function.
        
    Methods
    -------
    calculate(t)
        Computes the power spectral density function on an array of frequencies :math:`f`.
    
    """
    expression = 'multiplebendingpowerlaw'
    parameters: ParametersModel    
    N : int
    
    def __init__(self, parameters_values, **kwargs):     
        assert len(parameters_values) %2 == 1 and len(parameters_values)>=4 , f'The number of parameters for {self.__classname__()} must be greater than 4 and even, not {len(parameters_values)}'
        self.N = len(parameters_values)//2-1
        free_parameters = kwargs.get('free_parameters', [True]*len(parameters_values))
        # initialise the parameters and check
        names=['amplitude', 'freq_0', 'index_0']
        
        [(names.append(f'freq_{i}'),names.append(f'index_{i}')) for i in range(1,1+self.N)]
        
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
        P = jnp.power( f / self.parameters[f'freq_0'].value , -self.parameters[f'index_0'].value )
        for i in range(1,1+self.N):
            P *=   1/(1 + jnp.power( f / self.parameters[f'freq_{i}'].value ,
                    self.parameters[f'index_{i}'].value-self.parameters[f'index_{i-1}'].value ) )
        # P /= ( jnp.trapz(P)*(f[1]-f[0]))
        return P* self.parameters[f'amplitude'].value
    
class DoubleBendingPowerLaw(PowerSpectralDensity):
    r""" Class for the Multiple bending power-law power spectral density.

    .. math:: :label: multiplebendplpsd     
    
       \mathcal{P}(f) =  \dfrac{A \left({f/}{f_0}\right)^{-\alpha_0}}{\displaystyle\prod_{i=1}^{N} \left(1+\left(\dfrac{f}{f_{b_i}}\right)^{\alpha_{i+1}-\alpha_{i}}\right)}
    
    with the amplitude :math:`A\ge 0`, the position :math:`f_0\ge 0` and the standard-deviation '`sigma`' :math:`\sigma>0`.
    
    The parameters are stored in the `parameters` attribute which is a :class:`~pioran.parameters.ParametersModel` object. 
    The values of the parameters can be accessed using the `parameters` attribute via three keys: '`position`', '`amplitude`' and '`sigma`'
    
    The power spectral density function is evaluated on an array of frequencies :math:`f` using the `calculate` method.
    
    
    Parameters
    ----------
    param_values : :obj:`list of float`
        Values of the parameters of the power spectral density function.
    **kwargs : :obj:`dict`        
        free_parameters: :obj:`list of bool`
            List of bool to indicate if the parameters are free or not.
            
    Attributes
    ----------
    parameters : :class:`~pioran.parameters.ParametersModel`
        Parameters of the power spectral density function.
        
    Methods
    -------
    calculate(t)
        Computes the power spectral density function on an array of frequencies :math:`f`.
    
    """
    expression = 'doublebendingpowerlaw'
    parameters: ParametersModel    
    N : int
    
    def __init__(self, parameters_values, **kwargs):     
        assert len(parameters_values)  == 6 , f'The number of parameters for {self.__classname__()} must be 6, not {len(parameters_values)}'
        self.N = len(parameters_values)//2-1
        free_parameters = kwargs.get('free_parameters', [True]*len(parameters_values))
        # initialise the parameters and check
        names=['amplitude', 'index_1', 'freq_1', 'delindex_2', 'delfreq_2', 'delindex_3']
                
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
        
        A,index_1,f_1,delindex_1,delf_1,delindex_2 = self.parameters['amplitude'].value, self.parameters['index_1'].value, self.parameters['freq_1'].value, self.parameters['delindex_2'].value, self.parameters['delfreq_2'].value, self.parameters['delindex_3'].value
        f_2 = f_1*delf_1
        index_2 = index_1 + delindex_1
        index_3 = index_2 + delindex_2
        P = jnp.power(f,-index_1) / (1+jnp.power(f/f_1,index_2)) / (1+jnp.power(f/f_2,index_3))
        # s = jnp.trapz(jnp.power(f,-index_1) / (1+jnp.power(f/f_1,index_2)) / (1+jnp.power(f/f_2,index_3)))
        return A*P
        # P = jnp.power( f , -self.parameters[f'index_1'].value )
        # P *=   1/(1 + jnp.power( f / self.parameters[f'freq_1'].value ,(selfself.parameters[f'2'].value ) )
        # return P* self.parameters[f'amplitude'].value
        
class BrokenPowerLaw(PowerSpectralDensity):
    expression = 'brokenpowerlaw'
    parameters: ParametersModel    
    N : int
    
    def __init__(self, parameters_values, **kwargs):     
        assert len(parameters_values)  == 3 , f'The number of parameters for {self.__classname__()} must be 4, not {len(parameters_values)}'
        self.N = len(parameters_values)//2-1
        free_parameters = kwargs.get('free_parameters', [True]*len(parameters_values))
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
        return jnp.where(f<f_1, jnp.power(f/f_1,-index_1), jnp.power(f/f_1,-index_2))