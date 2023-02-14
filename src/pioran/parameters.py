"""Classes for the parameters of covariance functions

"""
from dataclasses import dataclass
from .tools import check_instance, TYPE_NUMBER, TABLE_LENGTH, HEADER_PARAMETERS
from .parameter_base import Parameter

import jax.numpy as jnp
import equinox as eqx




class ParametersModel:
    """ Class to store the parameters of a model. 

    This object stores one or several :obj:`Parameter` objects of a model. The model can be 
    of type :obj:`CovarianceFunction` , :obj:`PowerSpectralDensity` or :obj:`PowerSpectralDensityComponent`.
    Initialised with a list of values for the parameters.
    
    
    Parameters
    ----------
    param_values : list of float or list of Parameter objects
        Values of the parameters.
    names : list of str
        Names of the parameters.
    **kwargs : dict
        boundaries : list of float or list of None
            Boundaries of the parameters.
        free_parameters : list of bool
            True if the parameter is free, False otherwise.
            
    Attributes
    ----------
    all : dict of Parameter objects
        Dictionary with the name of the parameter as key and the Parameter object as value.
    names : list of str
        Names of the parameters.
    values : list of float
        Values of the parameters.
    boundaries : list of (list of float or list of None)
        Boundaries of the parameters.
    free_parameters : list of bool
        True if the parameter is free, False otherwise.

    """
    names: list
    free_parameters: jnp.ndarray
    components: jnp.ndarray
    IDs: list
    relations: list
    values: jnp.ndarray
    hyperparameters: list
    _pars:list = None


    def __init__(self,param_names,param_values,free_parameters,IDs=None,hyperparameters=None,components=None,relations=None,**kwargs):
        """Constructor method for the ParametersModel class.

        """
        if '_pars' in kwargs.keys():
            self._pars = (kwargs['_pars'])
        else:
            if IDs is None:
                IDs = [i for i in range(1,1+len(param_names))]
            if hyperparameters is None:
                hyperparameters = [True for i in range(len(param_names))]
            if components is None:
                components = [1 for i in range(len(param_names))]
            if relations is None:
                relations = [None for i in range(len(param_names))]
 
            self._pars = [Parameter(param_names[i],param_values[i],free_parameters[i],IDs[i],hyperparameters[i],components[i]) for i in range(len(param_names))]


    def increment_component(self,increment):
        """ Increment the component number of all the parameters by a given value.

        Parameters
        ----------
        increment : int
            Value used to increase the component number of the parameters.
        """
        for i in range(len(self.components)):
            self._pars[i].component += increment
            
    def increment_IDs(self, increment: int):
        """ Increment the ID of all the parameters by a given value.

        Parameters
        ----------
        increment : int
            Value used to increase the ID of the parameters.
        """
        for i in range(len(self.IDs)):
            self._pars[i].ID += increment

    def append(self, name,value,free,ID=None,hyperparameter=True,component=None,relation=None):
        """ Add a parameter to the list of objects.
 

        Parameters
        ----------
        name : str
            Name of the parameter.
        value : float
            Value of the parameter.
        free : bool
            True if the parameter is free, False otherwise.
        ID : int, optional
            ID of the parameter.
        hyperparameter : bool, optional
            True if the parameter is a hyperparameter, False otherwise.
        component : int, optional
            Component number of the parameter.
        relation : str, optional
            Relation between the parameter and the hyperparameters.
        """
        if ID is None:
            ID = len(self.IDs)+1
        self._pars.append(Parameter(name,value,free,ID,hyperparameter,component,relation))

    @property
    def names(self):
        """ Names of the parameters. 

        Returns
        -------
        list of str
            Names of the parameters.
        """
        return [P.name for P in self._pars]

    @property
    def free_parameters(self):
        """ Get the values of the free parameters.

        Returns
        -------
        values : list of float
            Values of the free parameters.
        """
        return [P.free for P in self._pars]
    
    @property
    def IDs(self):
        """ Get the ID of the parameters.

        Returns
        -------
        IDs : list of float
            IDs of the parameters.
        """
        return [P.ID for P in self._pars]
    
    @property
    def hyperparameters(self):
        return [P.hyperparameter for P in self._pars]
    
    @property
    def relations(self):
        return [P.relation for P in self._pars]
    
    @property
    def components(self):
        return [P.component for P in self._pars]

    def set_names(self,new_names):
        """ Set the names of the parameters. 

        Parameters
        ----------
        new_names : list of str
            New names of the parameters.

        Raises
        ------
        ValueError
            When the number of new names is not the same as the number of parameters.
        """
        assert len(new_names) == len(self.names), f"The number of new names ({len(new_names)}) is not the same as the number of parameters ({len(self.names)})."
        for i in range(len(self._pars)):
            self._pars[i].name = new_names[i]

    @property
    def values(self):
        """ Get the values of the parameters.

        Returns
        -------
        values : list of float
            Values of the parameters.
        """
        return [P.value for P in self._pars]

    @property
    def free_values(self):
        return [p.value for p in self.all.values() if p.free]
    
    def set_free_values(self,new_free_values):
        """ Set the values of the free parameters.

        Parameters
        ----------
        new_free_values : list of float
            Values of the free parameters.

        Raises
        ------
        ValueError
            When the number of new values is not the same as the number of free parameters.
        """
        assert len(new_free_values) == sum(self.free_parameters), f"The number of free parameters ({len(new_free_values)}) is not the same as the number of parameters ({sum(self.free_parameters)})."

        w = 0 
        for i in range(len(self._pars)):
            if self._pars[i].free:
                self._pars[i].value = new_free_values[w]
                w += 1


    def __getitem__(self, key):
        """ Get a Parameter object using the name of the parameter in square brackets or the index of the parameter in brackets.
        
        Get the parameter object with the name in brackets : ['name'] or the parameter object with the index in brackets : [index].
        If several parameters have the same name, the only the first one is returned.

        Parameters
        ----------
        key : str
            Name of the parameter.

        Returns
        -------
        parameter : Parameter object
            Parameter with name "key".

        Raises
        ------
        KeyError
            When the parameter is not in the list of parameters.
            
        """
        if key in self.names:
            return self._pars[self.names.index(key)]
        elif isinstance(key,int):
            return self._pars[key]
        else:
            raise KeyError(f"Parameter {key} not found.")
            

    def __setitem__(self, key, value : Parameter):
        """ Set a Parameter object using the name of the parameter in square brackets.

        Parameters
        ----------
        key : str
            Name of the parameter.
        value  : Parameter
            Value of the parameter with name "key".

        """
        
        if key in self.names:
            self._pars[self.names.index(key)].value = value
        else:
            raise KeyError(f"Parameter {key} not found.")
    

    def __str__(self) -> str:
        """ String representation of the Parameters object.
        
        

        Returns
        -------
        str 
            Pretty table with the info on all parameters.
            
        """
        s = ""+TABLE_LENGTH*"="+"\n"
        s += HEADER_PARAMETERS.format(Component='CID',ID='ID',Name="Name", Value="Value", Status="Status", Linked="Linked", Type="Type")
        s += "\n"
        for p in self._pars:
            s += str(p) + '\n'
        return s+"\n"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def __repr_html__(self) -> str:
        return self.__str__()