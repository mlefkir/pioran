from dataclasses import dataclass
from copy import deepcopy
import re

import jax.numpy as jnp
from .parameters import ParametersModel


@dataclass
class PowerSpectralDensityComponent:
    """Class for the components of Power spectral density.
    
    This class is used to specify individual components of PSD model. 
    See psd.py for examples of PSD components.
    
    Attributes
    ----------
    parameters: ParametersModel
        Parameters of the component
    componentname: str
        Name of the component
    ID: int
        ID of the component
    
    Methods
    -------
    __init__
    __add__
    __mul__
    __str__
    
    """
    
    parameters: ParametersModel
    componentname: str
    ID: int

    def __init__(self, parameters_values, names, boundaries, free_parameters, **kwargs):
        """Constructor of the PowerSpectralDensityComponent class 
        
        Parameters
        ----------
        parameters_values
        
        """
        if isinstance(parameters_values, ParametersModel):
            self.parameters = parameters_values
        elif isinstance(parameters_values, list) or isinstance(parameters_values, jnp.ndarray):
            self.parameters = ParametersModel(parameters_values, names=names,
            boundaries=boundaries,free_parameters=free_parameters)
        else:
            raise TypeError("The parameters of the covariance function must be a list of floats or jnp.ndarray or a ParametersModel object.")
      
    
    def __add__(self, other):
        """Add two PSD components.

        Overload of the + operator for the PowerSpectralDensityComponent class.

        Parameters
        ----------
        other: PowerSpectralDensityComponent or PowerSpectralDensity
            PowerS pectral Density to add.

        Returns
        -------
        PowerSpectralDensity
            Sum of the two PowerSpectralDensityComponents.

        Raises
        ------
        TypeError
            If the other parameter is not a Parameter or a number.
        """
        # make a copy of the other PSD and delete the original
        new = deepcopy(other)
        del other
        

         # if other is a PSD, then add the components of the other PSD to the list of components of the new PSD
        if isinstance(new, PowerSpectralDensityComponent) :
             # if other is a PSD component, then add it to the list of components of a new PSD
            new.ID = self.ID + 1
            print(f"new ID {new.ID}")
            new.parameters.increment_IDs(len(self.parameters))
            return PowerSpectralDensity(value = SumPSD(self, new), components= [self,new], expression = f"[{self.ID}]+[{new.ID}]")
        
        elif isinstance(new, PowerSpectralDensity):
            
            for newcomp in new.components:
                newcomp.parameters.increment_IDs(len(self.parameters))
                newcomp.ID += self.ID
            old_componentsID = re.findall(r'(\d+)',new.expression)
            old_componentsID.reverse()
            for k in old_componentsID:
                new.expression = new.expression.replace(k,str(int(k)+1))
            # new.parameters.increment_IDs(len(self.parameters))
            return PowerSpectralDensity(value = SumPSD(self, new), components= [self]+new.components, expression = f"[{self.ID}]+{new.expression}")
        else:
            raise TypeError(f"Cannot add a PSD component and a {type(new)}")


    def __mul__(self, other):
        """Multiply two PSD components.

        Overload of the * operator for the PowerSpectralDensityComponent class.

        Parameters
        ----------
        other: PowerSpectralDensityComponent or TYPE_NUMBER
            Parameter or number to multiply.

        Returns
        -------
        Parameters
            Sum of the two parameters.

        Raises
        ------
        TypeError
            If the other parameter is not a Parameter or a number.
        """
        # make a copy of the other PSD and delete the original
        new = deepcopy(other)
        del other
        
         # if other is a PSD, then add the components of the other PSD to the list of components of the new PSD
        if isinstance(new, PowerSpectralDensityComponent) :
            new.ID = self.ID + 1
            new.parameters.increment_IDs(len(self.parameters))
            return PowerSpectralDensity(value = ProductPSD(self, new), components= [self,new], expression = f"[{self.ID}]*[{new.ID}]")
               
        elif isinstance(new, PowerSpectralDensity):
            
            for newcomp in new.components:
                newcomp.parameters.increment_IDs(len(self.parameters))
                newcomp.ID += self.ID
            old_componentsID = re.findall(r'(\d+)',new.expression)
            old_componentsID.reverse()
            for k in old_componentsID:
                new.expression = new.expression.replace(k,str(int(k)+1))
            # new.parameters.increment_IDs(len(self.parameters))
            return PowerSpectralDensity(value = ProductPSD(self, new), components= [self]+new.components, expression = f"[{self.ID}]*({new.expression})")
        else:
            raise TypeError(f"Cannot multiply a PSD component and a {type(new)}")


    def __str__(self) -> str:
        """Return a string representation of the PSD component

        Returns
        -------
        str
            A string representation of the PSD component.
        """
        
        s =  f"Component [{self.ID}]: {self.componentname}"+"\n"
        # s += "\n"+TABLE_LENGTH*">"+"\n"
        s += self.parameters.__str__()
        return s

    @classmethod
    def __classname__(cls):
        """Return the name of the class.
        
        Returns
        -------
        str
            Name of the class.
        """
        return cls.__name__

    __radd__ = __add__
    __rmul__ = __mul__



@dataclass
class PowerSpectralDensity:
    """ Master class for the power density function functions.
    
    """
    model_expression:str 
    expression: str
    components: list[PowerSpectralDensityComponent]
    componentsID: list[int]
    
        
    def __init__(self, value,components: list[PowerSpectralDensityComponent],expression: str):
        """Constructor
        
        """
        self._value = value
        self.components = components   
        self.expression = expression
        self.componentsID = [comp.ID for comp in components]
        self.model_expression = expression
        
        for comp in components:
            self.model_expression = self.model_expression.replace(f"[{comp.ID}]",f"{comp.componentname}[{comp.ID}]")
        
        # precompiled_model = self.update_compiled_expression()
    
    def __add__(self, other) -> 'PowerSpectralDensity':
        """Add two PSD models.
        
        Overload the + operator to add two PSD. Will update the expression of the PSD and the ID of the PSD components.

        Parameters
        ----------
        other : PowerSpectralDensityComponent or PowerSpectralDensity
            The other PSD component to add to the current one.

        Returns
        -------
        PowerSpectralDensity
            The sum of the two PSD models.

        Raises
        ------
        TypeError
            If the other PSD component is not a PowerSpectralDensityComponent or a PowerSpectralDensity.
        """
        
        new = deepcopy(other)
        del other
   
        
        # if other is a PSD component, then convert it to a PSD
        if isinstance(new, PowerSpectralDensity):
            for comp in reversed(new.components): # reverse order to avoid changing the IDs
                comp.ID += len(self.components)    # increment the ID of the components
                comp.parameters.increment_IDs(sum([self.components[i].n_parameters for i in range(len(self.components))]))

            old_componentsID = re.findall(r'(\d+)',new.expression)
            old_componentsID.reverse()
            for k in old_componentsID:
                new.expression = new.expression.replace(k,str(int(k)+len(self.components) ))
            print(new.expression)

            self._value = SumPSD(self._value, new._value)
            self.expression = f"({self.expression}) + {new.expression}"
            self.components += new.components
            self.componentsID = [comp.ID for comp in self.components]
            self.model_expression = self.expression
            for comp in self.components:
                    self.model_expression = self.model_expression.replace(f"[{comp.ID}]",f"{comp.componentname}[{comp.ID}]")
            return self
        elif isinstance(new, PowerSpectralDensityComponent):
            return new.__add__(self)
        else:
            raise TypeError("Can only add PSD components or PSD objects")   
    
    def __mul__(self, other) -> 'PowerSpectralDensity':
        """Multiply two PSD models.
        
        Overload the * operator to multiply two PSD. Will update the expression of the PSD and the ID of the PSD components.

        Parameters
        ----------
        other : PowerSpectralDensityComponent or PowerSpectralDensity
            The other PSD component to add to the current one.

        Returns
        -------
        PowerSpectralDensity
            The product of the two PSD models.

        Raises
        ------
        TypeError
            If the other PSD component is not a PowerSpectralDensityComponent or a PowerSpectralDensity.
        """
        
        new = deepcopy(other)
        del other
   
        
        # if other is a PSD component, then convert it to a PSD
        if isinstance(new, PowerSpectralDensity):
            for comp in reversed(new.components): # reverse order to avoid changing the IDs
                comp.ID += len(self.components)    # increment the ID of the components
                comp.parameters.increment_IDs(sum([self.components[i].n_parameters for i in range(len(self.components))]))

            old_componentsID = re.findall(r'(\d+)',new.expression)
            old_componentsID.reverse()
            for k in old_componentsID:
                new.expression = new.expression.replace(k,str(int(k)+len(self.components) ))


            self._value = ProductPSD(self._value, new._value)
            self.expression = f"({self.expression}) * {new.expression}"
            self.components += new.components
            self.componentsID = [comp.ID for comp in self.components]
            self.model_expression = self.expression
            for comp in self.components:
                    self.model_expression = self.model_expression.replace(f"[{comp.ID}]",f"{comp.componentname}[{comp.ID}]")
            return self
        else:
            raise TypeError("Can only add PSD components or PSD objects")   
 
    def calculate(self, x: jnp.ndarray) -> jnp.ndarray:
        return self._value.calculate(x)
    
    
    def __str__(self) -> str:
        """String representation of the PSD
        
        Returns
        -------
        str
            String representation of the PSD.
        """
        # s = TABLE_LENGTH*"."
        s = ''
        for comp in self.components:
            s += comp.__str__()
            # s += TABLE_LENGTH*"."
        return s
    
        
    def __getitem__(self, key: int) -> PowerSpectralDensityComponent:
        """Return the component with the given ID

        Overload the [] operator to return the component with the given ID.

        Parameters
        ----------
        key : int
            The ID of the component to return.

        Returns
        -------
        PowerSpectralDensityComponent
            The component with the given ID.
        
        Raises
        ------
        ValueError
            If the ID is not in the list of components.
        """
        assert type(key) == int, "The key must be an integer"
        try:
            return self.components[self.componentsID.index(key)]
        except ValueError:
            raise ValueError(f"No component with ID {key} in the PSD.")

    # __rmul__ = __mul__


@dataclass
class SumPSD(PowerSpectralDensityComponent):
    """Base class for the sum of parameters.

    Attributes
    ----------
    first: Parameter
        First parameter to sum.
    second: Parameter
        Second parameter to sum.
    is_scalar: bool
        True if the second parameter is a scalar, False otherwise.
    fullname: str
        Full name of the resulting parameter.

    Methods
    -------
    value
        Returns the value of the resulting parameter.
    """

    def __init__(self, first: PowerSpectralDensityComponent, second: PowerSpectralDensityComponent, is_scalar=False):
        """Constructor of the class.

        Parameters
        ----------
        first: Parameter
            First parameter.
        second: Parameter
            Second parameter.
        is_scalar: bool
            True if the second parameter is a scalar, False otherwise.
        """

        self.first = first
        self.second = second

    def calculate(self, x: jnp.ndarray) -> jnp.ndarray:
        """Return the values of the resulting PSD.

        Parameters
        ----------
        x: np.ndarray
            Array of frequencies.

        Returns
        -------
        np.ndarray
            Array of values of the PSD SUM.
        """

        return self.first.calculate(x) + self.second.calculate(x)
    
    

@dataclass
class ProductPSD(PowerSpectralDensityComponent):
    """Class for the product of PowerSpectralDensityComponent.
    
    

    Attributes
    ----------
    first: PowerSpectralDensityComponent
        First parameter.
    second: PowerSpectralDensityComponent
        Second parameter to sum.
    is_scalar: bool
        True if the second parameter is a scalar, False otherwise.
    fullname: str
        Full name of the resulting parameter.

    Methods
    -------
    value
        Returns the value of the resulting parameter.
    """

    def __init__(self, first: PowerSpectralDensityComponent, second: PowerSpectralDensityComponent, is_scalar=False):
        """Constructor of the class.

        Parameters
        ----------
        first: Parameter
            First parameter.
        second: Parameter
            Second parameter.
        is_scalar: bool
            True if the second parameter is a scalar, False otherwise.
        """

        self.first = first
        self.second = second

    def calculate(self, x: jnp.ndarray) -> jnp.ndarray:
        """Return the values of the resulting PSD.

        Parameters
        ----------
        x: np.ndarray
            Array of frequencies.

        Returns
        -------
        np.ndarray
            Array of values of the PSD SUM.
        """

        return self.first.calculate(x) * self.second.calculate(x)
        