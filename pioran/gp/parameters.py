"""Classes for the parameters of covariance functions

"""
import numpy as np
from .tools import check_instance

TYPE_NUMBER = (float,int,np.number)


class Parameter:
    """Class for one parameters of a GP, it can be a hyperparameter or a mean parameter.
    
    Attributes
    ----------
    name : str
        Name of the parameter.
    value : float
        Value of the parameter.
    bounds : tuple
        Bounds of the parameter.
    free : bool
        If the parameter is free or fixed.


    Methods
    -------
    __str__(self)
        Print the parameters.
    
    """
    
    def __init__(self, name: str, value: float, bounds: list, free: bool = True):
        """Constructor method for the Parameter class.
        
        Parameters
        ----------
        name : str
            Name of the parameter.
        value : float
            Value of the parameter.
        bounds : tuple
            Bounds of the parameter.
        free : bool
            If the parameter is free or fixed.
        
        """
        self.name = name
        self.value = value
        self.bounds = bounds
        self.free = free
        
    
    def __str__(self):
        """Print the parameter in a pretty formatting.

        """
        bnd = lambda x : f"{self.bounds[x]:3.2e}" if self.bounds[x] is not None else " ... "
        return f" {self.name:<15}{self.value:4e}\t({bnd(0)} , {bnd(1)})\t{'Free' if self.free else 'Fixed'}"


class ParametersCovFunction(dict):
    """ Class for the parameters of a covariance function. 

    Initialised with a list of values for the parameters.


    Attributes
    ----------
    all : dict of Parameter objects
        Dictionary with the name of the parameter as key and the Parameter object as value.
    names : list of str
        Names of the parameters.
    values : list of float
        Values of the parameters.
    boundaries : list of tuples
        Boundaries of the parameters.
    free_parameters : list of bool
        True if the parameter is free, False otherwise.

    Methods
    -------
    update_names
        Update the parameters names.
    update_boundaries
        Update the boundaries of the parameters.
    add_parameter
        Add a parameter to the object.
    check_boundaries
        Check if the parameters are within the boundaries.
    print_parameters
        Print the parameters.
    __getitem__
        Get the value of a parameter using the name of the parameter in square brackets.

    """

    def __init__(self, param_values, names ,**kwargs):
        """Constructor method for the ParametersCovFunction class.

        Parameters
        ----------
        param_values : list of float or list of Parameter objects
            Values of the parameters.
        names : list of str
            Names of the parameters.
        **kwargs : dict
            boundaries : list of float or list of None
                Boundaries of the parameters.
        
        Raises
        ------
        ValueError
            When the number of parameters or boundaries is not the same as the number of names.

        """
        # sanity checks
        assert len(param_values) == len(names), "The number of parameters is not the same as the number of names."
        if "boundaries" in kwargs.keys():
            assert len(kwargs["boundaries"]) == len(names), "The number of boundaries is not the same as the number of names."
            boundaries = kwargs["boundaries"]
            
        # check if the parameters are given as a list of Parameter objects
        if check_instance(param_values, Parameter):
            self.all = dict(zip([p.name for p in param_values],param_values))
        elif check_instance(param_values, TYPE_NUMBER):
            if "boundaries" in kwargs.keys():
                self.all = {key: Parameter(name=key, value=value, bounds=bounds) for (key, value,bounds) in zip(names, param_values, boundaries)}
            else :
                self.all = {key: Parameter(name=key, value=value, bounds=[None,None]) for (key, value) in zip(names, param_values)}

        self.names = names
        self.values = self.all.values() 
        self.free_parameters = [p.free for p in self.all.values()]
        self.boundaries = [p.bounds for p in self.all.values()]

    def append(self, parameter: Parameter):
        """ Add a parameter to the list of objects.

        Parameters
        ----------
        parameter : Parameter object
            Parameter to add to the object.

        """
        self.all[parameter.name] = parameter
        
    def __len__(self) -> int:
        """Length of the object."""
        return len(self.all)
    
    @property
    def boundaries(self):
        """Get the boundaries of the parameters.

        Returns
        -------
        boundaries : list of tuples
            Boundaries of the parameters.
        """
        # update the list of boundaries if the boundaries of the parameters have changed
        self._boundaries = [p.bounds for p in self.all.values()]
        return [p.bounds for p in self.all.values()]
    
    @boundaries.setter
    def boundaries(self, new_boundaries):
        """Set the boundaries of the parameters.

        Parameters
        ----------
        boundaries : list of (list of float or list of None)
            Boundaries of the parameters.

        """
        assert len(new_boundaries) == len(self.all), "The number of boundaries is not the same as the number of parameters."
        # also update the boundaries of the parameters
        for i, b in enumerate(new_boundaries):
            assert len(b) == 2, "The boundaries must be a list of 2 elements."
            
            # check the boundaries
            if (b[0] is not None and  b[1] is not None):
                assert b[0] < b[1], "The lower boundary must be smaller than the upper boundary."
            else :
                assert b[0] is None or b[1] is None, "The boundaries must be None or a number."            
            self.all[self.names[i]].bounds = b
            
        self._boundaries = new_boundaries


    @property
    def names(self):
        """ Names of the parameters. 
        
        Returns
        -------
        list of str
            Names of the parameters.
        """ 
        self._names = [p.name for p in self.all.values()]
        return self._names
    
    @names.setter
    def names(self, new_names):
        """ Set the names of the parameters. 
        
        Parameters
        ----------
        new_names : list of str
            New names of the parameters.
            
        Raises
        ------
        ValueError
            When the number of new names is not the same as the number of parameters.
        TypeError
            When the new names are not a list of strings.
        """
        if len(new_names) == len(self.all):
            if check_instance(new_names, str):
                self._names = new_names
                # also update the names of the parameters of Parameter objects
                for i, p in enumerate(self.all.values()):
                    p.name = new_names[i]
            else:
                raise TypeError("The names must be a list of strings.")
        else:
            raise ValueError("The number of names is not the same as the number of parameters.")

    @property
    def values(self):
        """ Get the values of the parameters.

        Returns
        -------
        values : list of float
            Values of the parameters.
        """
        # update the values in case they have changed since the last call
        self.values = self.all.values()
        return self._values
    
    @values.setter
    def values(self, new_values):
        """ Set the values of the parameters.

        Parameters
        ----------
        values : list of float or list of Parameter objects
            Values of the parameters.
        """
        if len(new_values) == len(self.all):
            if check_instance(new_values, Parameter):
                self._values = [p.value for p in new_values]
            elif check_instance(new_values, TYPE_NUMBER):
                self._values = new_values
                # also update the values of the parameters of the Parameter objects
                for i, p in enumerate(self.all.values()):
                    p.value = new_values[i]
            else:
                raise TypeError("The values must be a list of numbers or a list of Parameter objects.")
        else:
            raise ValueError(f"The number of values ({len(new_values)}) is not the same as the number of parameters ({len(self.all)}). ")

    @property
    def free_parameters(self):
        """ Get the list of bool, True if the parameter is free, False otherwise. 
        
        Returns
        -------
        list of bool
            True if the parameter is free, False otherwise.
        """
        # update the values in case they have changed since the last call
        self.free_parameters = [p.free for p in self.all.values()]
        return self._free_parameters
    
    @free_parameters.setter
    def free_parameters(self, new_free_parameters):
        """ Set the free parameters.

        Parameters
        ----------
        free_parameters : list of bool
            True if the parameter is free, False otherwise.
        """
        assert len(new_free_parameters) == len(self.all), "The number of free parameters is not the same as the number of parameters."
        assert check_instance(new_free_parameters, bool), "The free parameters must be a list of booleans."
        self._free_parameters = new_free_parameters
        # also update the free parameters of the Parameter objects
        for i, p in enumerate(self.all.values()):
            p.free = new_free_parameters[i]
    
    def __getitem__(self, key):
        """ Get a Parameter object using the name of the parameter in square brackets.

        Parameters
        ----------
        key : str
            Name of the parameter.

        Returns
        -------
        parameter : Parameter object
            Parameter with name "key".
        """
        if key in self.all.keys():
            return self.all[key]
        else:
            raise KeyError(f"Parameter {key} not found.")
        
    def __setitem__(self, key, value: Parameter):
        """ Set a Parameter object using the name of the parameter in square brackets.

        Parameters
        ----------
        key : str
            Name of the parameter.
        value : Parameter object
            Value of the parameter with name "key".

        """
        self.all[key] = value
        self.values = self.all.values()
        self.free_parameters = [p.free for p in self.all.values()]

    def __str__(self):
        """ Print the parameters. """
        s = f"=======================================================\n"

        s += f" name\t       value\t\t( min , max )\tstatus\n"
        s +=f"-------------------------------------------------------\n"
        for p in self.all.values():
            s += p.__str__() + "\n"
        s +=f"======================================================="
        return s
    
        
    # def update_boundaries(self, boundaries):
    #     """ Update the boundaries of the parameters

    #     Parameters
    #     ----------
    #     boundaries : list of tuples
    #     """

    #     assert len(boundaries) == len(
    #         self.values), "The number of boundaries is not the same as the number of parameters."
    #     self.boundaries = {}

    #     for i, b in enumerate(boundaries):
    #         assert len(b) == 2, "The boundaries must be a list of 2 elements."
    #         self.boundaries[self.names[i]] = b

    # def add_parameter(self, name, value, boundaries):
    #     """ Add a parameter to the object.

    #     Parameters
    #     ----------
    #     name : str
    #         Name of the parameter.
    #     value : float
    #         Value of the parameter.
    #     boundaries : tuple
    #         Boundaries of the parameter.
    #     """
    #     self.names.append(name)
    #     self.values.append(value)
    #     self.boundaries[name] = boundaries


    # def check_boundaries(self):
    #     """ Check if the parameters are within the boundaries. 

    #     Returns
    #     -------
    #     bool : True if the parameters are within the boundaries, False otherwise.
    #     """
    #     for i in range(len(self.names)):
    #         if self.values[i] < self.boundaries[self.names[i]][0] or self.values[i] > self.boundaries[self.names[i]][1]:
    #             return False
    #     return True


