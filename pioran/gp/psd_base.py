from dataclasses import dataclass
from copy import deepcopy
import types

import jax.numpy as jnp
from .parameters import ParametersModel


@dataclass
class PowerSpectralDensityComponent:
    parameters: ParametersModel
    componentname: str
    ID: int

    def __init__(self, parameters_values, names, boundaries, free_parameters, **kwargs):
        """Constructor
        
        """
        if isinstance(parameters_values, ParametersModel):
            self.parameters = parameters_values
        elif isinstance(parameters_values, list) or isinstance(parameters_values, jnp.ndarray):
            self.parameters = ParametersModel(parameters_values, names=names,
            boundaries=boundaries,free_parameters=free_parameters)
        else:
            raise TypeError("The parameters of the covariance function must be a list of floats or jnp.ndarray or a ParametersModel object.")

    
    def __add__(self, other) -> 'PowerSpectralDensity':
        """Add two PSD components
        
        Overload the + operator to add two PSD components. Will update the expression of the PSD and the ID of the PSD components.

        Parameters
        ----------
        other : PowerSpectralDensityComponent or PowerSpectralDensity
            The other PSD component to add to the current one.

        Returns
        -------
        PowerSpectralDensity
            The sum of the two PSD components.

        Raises
        ------
        TypeError
            If the other PSD component is not a PowerSpectralDensityComponent or a PowerSpectralDensity.
        """   
        
        # make a copy of the other PSD and delete the original
        new = deepcopy(other)
        del other
        
        # if other is a PSD component, then add it to the list of components of a new PSD
        if isinstance(new, PowerSpectralDensityComponent):
            new.ID = self.ID + 1
            new.parameters.increment_IDs(len(self.parameters))
            return PowerSpectralDensity([self,new], expression = f"[{self.ID}]+[{new.ID}]")
        # if other is a PSD, then add the components of the other PSD to the list of components of the new PSD
        elif isinstance(new, PowerSpectralDensity):
            for comp in reversed(new.components): # reverse order to avoid changing the IDs 
                new.expression = new.expression.replace(f"[{comp.ID}]",f"[{comp.ID+1}]") # change the IDs in the expression
                comp.ID += 1 # increment the ID of the component
            return PowerSpectralDensity([self]+new.components, expression = f"[{self.ID}]+{new.expression}")
        else:
            raise TypeError("Can only sum PSD components or PSD objects")
      
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
        
    def __call__(self) -> 'PowerSpectralDensity':
        """Return a PowerSpectralDensity object

        Calling a PSD component returns a PSD object with the component as the only component.

        Returns
        -------
        PowerSpectralDensity
            A PSD object with the component as the only component.
        """
        
        return PowerSpectralDensity([self], expression = '[1]')

    
    def __mul__(self, other) -> 'PowerSpectralDensity':
        """Multiply two PSD components
        
        Overload the * operator to multiply two PSD components. Will update the expression of the PSD and the ID of the PSD components.


        Parameters
        ----------
        other : PowerSpectralDensityComponent or PowerSpectralDensity
            The other PSD component to add to the current one.

        Returns
        -------
        PowerSpectralDensity
            The sum of the two PSD components.

        Raises
        ------
        TypeError
            If the other PSD component is not a PowerSpectralDensityComponent or a PowerSpectralDensity.
        """
        
        # make a copy of the other PSD and delete the original
        new = deepcopy(other)
        del other
        
        # if other is a PSD component, then add it to the list of components of a new PSD
        if isinstance(new, PowerSpectralDensityComponent):
            new.ID = self.ID + 1
            new.parameters.increment_IDs(len(self.parameters))
            return PowerSpectralDensity([self,new], expression = f"[{self.ID}]*[{new.ID}]")
        elif isinstance(new, PowerSpectralDensity):
            for comp in reversed(new.components): # reverse order to avoid changing the IDs
                new.expression = new.expression.replace(f"[{comp.ID}]",f"[{comp.ID+1}]") # change the IDs in the expression
                comp.ID += 1    # increment the ID of the component
            return PowerSpectralDensity([self]+new.components, expression = f"[{self.ID}]*({new.expression})")
        else:
            raise TypeError("Can only multiply PSD components or PSD objects")  
    
    def __eq__(self, other) -> bool:
        """Check if two PSD components are equal
        
        Overload the == operator to check if two PSD components are equal.

        Parameters
        ----------
        __o : object
            The other PSD component to compare to the current one.

        Returns
        -------
        bool
            True if the two PSD components are equal, False otherwise.
        """
        
        if isinstance(other, PowerSpectralDensityComponent):
            return self.parameters == other.parameters and self.componentname == other.componentname
        else:
            return False
    
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
        
    def __init__(self, components: list[PowerSpectralDensityComponent],expression: str):
        """Constructor
        
        """
        self.components = components   
        self.expression = expression
        self.componentsID = [comp.ID for comp in components]
        self.model_expression = expression
        
        for comp in components:
            self.model_expression = self.model_expression.replace(f"[{comp.ID}]",f"{comp.componentname}[{comp.ID}]")
        
        precompiled_model = self.update_compiled_expression()
        
        
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
    
    
    def update_compiled_expression(self):
        
        precompiled_model = self.expression
        
        for comp in self.components:
            # create a precompiled model with constants already replaced by their values
            if comp.componentname == 'scalar':
                precompiled_model = precompiled_model.replace(f"[{comp.ID}]",f"{comp.fun(0)}") # value of the scalar function

        model = f"""def evaluate(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
            return ({precompiled_model.replace('[','self.__getitem__(').replace(']',').fun(x)')})"""
        # model = f"""def evaluate(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
                # return ({expression.replace('[','self.__getitem__(').replace(']',').fun(x)')})"""
        compiled_model = compile(model, '<string>', 'exec')
        exec(compiled_model)
        self.evaluate = types.MethodType(locals()['evaluate'],self)
        return precompiled_model
    
    
    # def evaluate(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
    #     """Evaluate the power spectral density function at a given frequency.
        
    #     Parameters
    #     ----------
    #     x : jnp.ndarray
    #         Frequency at which to evaluate the power spectral density function.
    #     **kwargs : 
    #         Parameters of the power spectral density function.

    #     Returns
    #     -------
    #     jnp.ndarray
    #         Power spectral density function evaluated at the given frequency.
    #     """
    #     precompiled_model = self.update_compiled_expression()
    #     return eval(f"{precompiled_model.replace('[','self.__getitem__(').replace(']',').fun(x)')}")
        
        
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
        if isinstance(new, PowerSpectralDensityComponent):
            new = new.__call__()
        
        if isinstance(new, PowerSpectralDensity):
            
            indexes = [int(char) for char in new.expression if not char in ['+',"*","[",']'] ]  # find the IDs in the other PSD
            startIndex = max([int(char) for char in self.expression if not char in ['+',"*","[",']'] ]) # find the highest ID in the current PSD
            
            for i in reversed(indexes): # reverse order to avoid changing the IDs 
                new.expression = new.expression.replace(f"[{new[i].ID}]",f"[{new[i].ID+startIndex}]") # change the IDs in the expression
                new[i].ID += startIndex # increment the ID of the component
            return PowerSpectralDensity(self.components+new.components, expression = f"{self.expression}+{new.expression}")
        else:
            raise TypeError("Can only add PSD components or PSD objects")   
    
    def __mul__(self, other) -> 'PowerSpectralDensity':
        """Multiply two PSD models.

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
        
        if isinstance(new, PowerSpectralDensityComponent):
            # return new * self
            new = new.__call__()
        
        if isinstance(new, PowerSpectralDensity):
            indexes = [int(char) for char in new.expression if not char in ['+',"*","[",']'] ]  # find the IDs in the other PSD
            startIndex = max([int(char) for char in self.expression if not char in ['+',"*","[",']'] ]) # find the highest ID in the current PSD
            
            for i in reversed(indexes): # reverse order to avoid changing the IDs 
                new.expression = new.expression.replace(f"[{new[i].ID}]",f"[{new[i].ID+startIndex}]") # change the IDs in the expression
                new[i].ID += startIndex # increment the ID of the component
            
            a_expression = f"({self.expression})" if len(self.components) > 1 else self.expression
            b_expression = f"({new.expression})" if len(new.components) > 1 else new.expression
            return PowerSpectralDensity(self.components+new.components, expression = f"{a_expression}*{b_expression}")
        else:
            raise TypeError("Can only multiply PSD components or PSD objects")    
    
    __radd__ = __add__
    __rmul__ = __mul__


