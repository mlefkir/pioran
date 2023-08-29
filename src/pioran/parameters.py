"""Classes to store for the parameters of models.

"""
import warnings

import jax.numpy as jnp
from .parameter_base import Parameter
from .tools import HEADER_PARAMETERS, TABLE_LENGTH


class ParametersModel:
    """ Class to store the parameters of a model. 

    This object stores one or several :class:`~pioran.parameter_base.Parameter` objects for a model. The model can be 
    of type :class:`~pioran.acvf_base.CovarianceFunction`  or :class:`~pioran.psd_base.PowerSpectralDensity`.
    Initialised with a list of values for the parameters.
    
    
    Parameters
    ----------
    param_names : :obj:`list` of :obj:`str`
        Names of the parameters.
    param_values : :obj:`list` of :obj:`float`
        Values of the parameters.
    free_parameters : :obj:`list` of :obj:`bool`, optional
        List of bool to indicate if the parameters are free or not.
    IDs : :obj:`list` of :obj:`int`, optional
        IDs of the parameters.
    hyperparameters : :obj:`list` of :obj:`bool`, optional
        List of bool to indicate if the parameters are hyperparameters or not.
    components : :obj:`list` of :obj:`int`, optional
        List of int to indicate the component number of the parameters.
    relations : :obj:`list` of :obj:`str`, optional
        List of str to indicate the relation between the parameters.
    kwargs : :obj:`dict`, optional
        _pars : :obj:`list` of :class:`~pioran.parameter_base.Parameter`
            List of Parameter objects.
            
    Attributes
    ----------
    names : :obj:`list` of :obj:`str`
        Names of the parameters.
    values : :obj:`list` of :obj:`float`
        Values of the parameters.
    free_parameters : :obj:`list` of :obj:`bool`
        True if the parameter is free, False otherwise.
    IDs : :obj:`list` of :obj:`int`
        IDs of the parameters.
    hyperparameters : :obj:`list` of :obj:`bool`
        True if the parameter is a hyperparameter, False otherwise.
    components : :obj:`list` of :obj:`int`
        Component number of the parameters.
    relations : :obj:`list` of :obj:`str`
        Relation between the parameters.
    _pars : :obj:`list` of :class:`~pioran.parameter_base.Parameter`
        List of Parameter objects.
    free_values : :obj:`list` of :obj:`float`
        Values of the free parameters.
        
    Methods
    -------
    increment_component(increment)
        Increment the component number of the parameters.
    increment_ID(increment)
        Increment the ID of the parameters.
    append(param)
        Append a parameter to the list of parameters.
    set_free_values(values)
        Set the values of the free parameters.
    set_names(names)
        Set the names of the parameters.
    __getitem__(key)
        Get the parameter at the index key.
    __setitem__(key,value)
        Set the parameter at the index key to value.
    __str__()
        String representation of the parameters.
    __repr__()
        Representation of the parameters.
    __len__()
        Number of the parameters.
    """
    names: list
    free_parameters: jnp.ndarray
    components: jnp.ndarray
    IDs: list
    relations: list
    values: jnp.ndarray
    hyperparameters: list
    _pars:list = None


    def __init__(self,param_names:list[str],param_values:list[float],free_parameters:list[bool],IDs:list[int]=None,hyperparameters:list[bool]=None,components:list[int]=None,relations:list[str]=None,**kwargs):
                #  ,param_values,free_parameters,IDs=None,hyperparameters=None,components=None,relations=None,**kwargs):
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


    def increment_component(self,increment:int):
        """ Increment the component number of all the parameters by a given value.

        Parameters
        ----------
        increment : :obj:`int`
            Value used to increase the component number of the parameters.
        """
        for i in range(len(self.components)):
            self._pars[i].component += increment
            
    def increment_IDs(self, increment: int):
        """ Increment the ID of all the parameters by a given value.

        Parameters
        ----------
        increment : :obj:`int`
            Value used to increase the ID of the parameters.
        """
        for i in range(len(self.IDs)):
            self._pars[i].ID += increment

    def append(self, name: str,value: float,free: bool,ID=None,hyperparameter=True,component=None,relation=None):
        """ Add a parameter to the list of objects.
 

        Parameters
        ----------
        name : :obj:`str`
            Name of the parameter.
        value : `float`
            Value of the parameter.
        free : :obj:`bool`
            True if the parameter is free, False otherwise.
        ID : :obj:`int`, optional
            ID of the parameter.
        hyperparameter : :obj:`bool`, optional
            True if the parameter is a hyperparameter, False otherwise. 
            The default is True.
        component : :obj:`int`, optional
            Component number of the parameter.
        relation : :obj:`str`, optional
            Relation between the parameters.
        """
        if ID is None:
            ID = len(self.IDs)+1
        if component is None:
            component = 1
        if name in self.names:
            warnings.warn(f"Parameter {name} already exists, the parameter should be accessed with the index.")
        self._pars.append(Parameter(name,value,free,ID,hyperparameter,component,relation))

    @property
    def names(self):
        """ Get the names of the parameters. 

        Returns
        -------
        :obj:`list of str`
            Names of the parameters.
        """
        return [P.name for P in self._pars]

    @property
    def free_parameters(self):
        """Get the values of the free parameters.

        Returns
        -------
        :obj:`list` of :obj:`float``
            Values of the free parameters.
        """
        return [P.free for P in self._pars]
    
    @property
    def IDs(self):
        """ Get the ID of the parameters.

        Returns
        -------
        :obj:`list` of :obj:`float``
            IDs of the parameters.
        """
        return [P.ID for P in self._pars]
    
    @property
    def hyperparameters(self):
        """ Get the hyperparameters of the parameters.
        
        Returns
        -------
        :obj:`list of bool`
            List of bool to indicate if the parameters are hyperparameters or not.
        """
        
        return [P.hyperparameter for P in self._pars]
    
    @property
    def relations(self):
        """Get the relation between the parameters.

        Returns
        -------
        :obj:`list of str`
            Relation between the parameters.
        """
        
        return [P.relation for P in self._pars]
    
    @property
    def components(self):
        """Get the component numbers of the parameters.

        Returns
        -------
        :obj:`list of int`
            Component numbers of the parameters.
        """
        
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
        :obj:`list` of :obj:`float``
            Values of the parameters.
        """
        return [P.value for P in self._pars]

    @property
    def free_values(self):
        """ Get the values of the free parameters.

        Returns
        -------
        :obj:`list` of :obj:`float``
            Values of the free parameters.
        """
        return [p.value for p in self._pars if p.free]
    
    @property
    def free_names(self):
        """ Get the names of the free parameters.

        Returns
        -------
        :obj:`list of str`
            Names of the free parameters.
        """
        
        return [p.name for p in self._pars if p.free]
    
    def set_free_values(self,new_free_values) -> None:
        """ Set the values of the free parameters.

        Parameters
        ----------
        new_free_values : :obj:`list` of :obj:`float``
            Values of the free parameters.

        Raises
        ------
        ValueError
            When the number of new values is not the same as the number of free parameters.
        """
        # assert len(new_free_values) == sum(self.free_parameters), f"The number of free parameters ({len(new_free_values)}) is not the same as the number of parameters ({sum(self.free_parameters)})."

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
        key : :obj:`str`
            Name of the parameter.

        Returns
        -------
        parameter : `Parameter` object
            Parameter with name "key".

        Raises
        ------
        KeyError
            When the parameter is not in the list of parameters.
            
        """
        if key in self.names:
            return self._pars[self.names.index(key)]
        
        elif isinstance(key,int):
            if key == 0:
                raise KeyError(f"Parameter at index 0 does not exist, use index 1 instead.")
            return self._pars[key-1]
        else:
            raise KeyError(f"Parameter {key} not found.")
    
    def __len__(self):
        """ Get the number of parameters.
        
        Returns
        -------
        :obj:`int`
            Number of parameters.        
        """
        return len(self._pars)

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
    

    def __str__(self) -> str: # pragma: no cover
        """ String representation of the Parameters object.
        
        Returns
        -------
        :obj:`str`
            Pretty table with the info on all parameters.
            
        """
        s = ""+TABLE_LENGTH*"="+"\n"
        s += HEADER_PARAMETERS.format(Component='CID',ID='ID',Name="Name", Value="Value", Status="Status", Linked="Linked", Type="Type")
        s += "\n"
        for p in self._pars:
            s += str(p) + '\n'
        s += f'\nNumber of free parameters : {sum(self.free_parameters)}\n'
        return s+"\n"
    
    def __repr__(self) -> str: # pragma: no cover
        return self.__str__()
    
    def __repr_html__(self) -> str: # pragma: no cover
        return self.__str__()