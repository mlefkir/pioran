"""Classes for the parameters of covariance functions

"""
from dataclasses import dataclass
from .tools import check_instance, TYPE_NUMBER, TABLE_LENGTH, HEADER_PARAMETERS
from .parameter_base import Parameter


@dataclass  # (slots=True)
class ParametersModel:
    """ Class for the parameters of a covariance function. 

    Initialised with a list of values for the parameters.


    Attributes
    ----------
    all: dict of Parameter objects
        Dictionary with the name of the parameter as key and the Parameter object as value.
    names: list of str
        Names of the parameters.
    values: list of float
        Values of the parameters.
    boundaries: list of (list of float or list of None)
        Boundaries of the parameters.
    free_parameters: list of bool
        True if the parameter is free, False otherwise.

    Methods
    -------
    update_names:
        Update the parameters names.
    update_boundaries:
        Update the boundaries of the parameters.
    add_parameter:
        Add a parameter to the object.
    check_boundaries:
        Check if the parameters are within the boundaries.
    print_parameters:
        Print the parameters.
    __getitem__:
        Get the value of a parameter using the name of the parameter in square brackets.

    """
    all: dict[str, Parameter]
    names: list[str]
    values: list[float]
    boundaries: list
    free_parameters: list[bool]


    def __init__(self, param_values, names, **kwargs):
        """Constructor method for the ParametersModel class.

        Parameters
        ----------
        param_values: list of float or list of Parameter objects
            Values of the parameters.
        names: list of str
            Names of the parameters.
        **kwargs: dict
            boundaries: list of float or list of None
                Boundaries of the parameters.
            free_parameters: list of bool
                True if the parameter is free, False otherwise.
        """
        # sanity checks
        assert len(param_values) == len(names), "The number of parameters is not the same as the number of names."
        if "boundaries" in kwargs.keys():
            assert len(kwargs["boundaries"]) == len(
                names), "The number of boundaries is not the same as the number of names."
            boundaries = kwargs["boundaries"]
        if "free_parameters" in kwargs.keys():
            assert len(kwargs["free_parameters"]) == len(
                names), "The number of free parameters is not the same as the number of names."
            free_parameters = kwargs["free_parameters"]

        # check if the parameters are given as a list of Parameter objects
        if check_instance(param_values, Parameter):
            self.all = dict(zip([p.name for p in param_values], param_values))
        elif check_instance(param_values, TYPE_NUMBER):
            self.all = {key: Parameter(name=key, value=value, bounds=[None, None],ID=counter+1) for counter,(
                key, value) in enumerate(zip(names, param_values))}
            if "boundaries" in kwargs.keys():
                for i, key in enumerate(self.all.keys()):
                    self.all[key].bounds = boundaries[i]
            if "free_parameters" in kwargs.keys():
                for i, key in enumerate(self.all.keys()):
                    self.all[key].free = free_parameters[i]

        self.names = names
        self.values = self.all.values()
        self.free_parameters = [p.free for p in self.all.values()]
        self.boundaries = [p.bounds for p in self.all.values()]
        

    def increment_IDs(self, increment: int):
        """ Increment the ID of all parameters by a given value.

        Parameters
        ----------
        increment: int
            Value to increase the ID of the parameters.
        """
        for p in self.all.values():
            p.ID += increment

    def append(self, parameter: Parameter):
        """ Add a parameter to the list of objects.

        Parameters
        ----------
        parameter: Parameter
            Parameter to add to the object.

        """
        self.all[parameter.name] = parameter

    def __eq__(self, other) -> bool:
        """Compare two parameters.
        """
        return self.all == other.all

    def __len__(self) -> int:
        """Length of the object.

        Returns
        -------
        int
            Total number of parameters.
        """
        return len(self.all)

    @property
    def boundaries(self):
        """Get the boundaries of the parameters.

        Returns
        -------
        boundaries: list of lists
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
        new_boundaries: list of (list of float or list of None)
            Boundaries of the parameters.
        """
        assert len(new_boundaries) == len(
            self.all), "The number of boundaries is not the same as the number of parameters."
        # also update the boundaries of the parameters
        for i, b in enumerate(new_boundaries):
            assert len(b) == 2, "The boundaries must be a list of 2 elements."

            # check the boundaries
            if (b[0] is not None and b[1] is not None):
                assert b[0] < b[1], "The lower boundary must be smaller than the upper boundary."
            else:
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
        new_names: list of str
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
            raise ValueError(
                "The number of names is not the same as the number of parameters.")

    @property
    def values(self):
        """ Get the values of the parameters.

        Returns
        -------
        values: list of float
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
        new_values: list of float or list of Parameter objects
            Values of the parameters.

        Raises
        ------
        TypeError
            When the new values are not a list of floats or Parameter objects.
        ValueError
            When the number of new values is not the same as the number of parameters.
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
                raise TypeError(
                    "The values must be a list of numbers or a list of Parameter objects.")
        else:
            raise ValueError(
                f"The number of values ({len(new_values)}) is not the same as the number of parameters ({len(self.all)}). ")

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
        new_free_parameters: list of bool
            True if the parameter is free, False otherwise.
        """
        assert len(new_free_parameters) == len(
            self.all), "The number of free parameters is not the same as the number of parameters."
        assert check_instance(
            new_free_parameters, bool), "The free parameters must be a list of booleans."
        self._free_parameters = new_free_parameters
        # also update the free parameters of the Parameter objects
        for i, p in enumerate(self.all.values()):
            p.free = new_free_parameters[i]

    def __getitem__(self, key):
        """ Get a Parameter object using the name of the parameter in square brackets.

        Parameters
        ----------
        key: str
            Name of the parameter.

        Returns
        -------
        parameter: Parameter object
            Parameter with name "key".

        Raises
        ------
        KeyError
            When the parameter is not in the list of parameters.
        """
        if key in self.all.keys():
            return self.all[key]
        else:
            raise KeyError(f"Parameter {key} not found.")

    def __setitem__(self, key, value: Parameter):
        """ Set a Parameter object using the name of the parameter in square brackets.

        Parameters
        ----------
        key: str
            Name of the parameter.
        value: Parameter
            Value of the parameter with name "key".

        """
        self.all[key] = value
        self.values = self.all.values()
        self.free_parameters = [p.free for p in self.all.values()]

    def __str__(self):
        """ String representation of the Parameters object.

        Returns
        -------
        str 
            Pretty table with the info on all parameters.
        """
        s = ""+TABLE_LENGTH*"="+"\n"
        s += HEADER_PARAMETERS.format(ID='ID',Name="Name", Value="Value", Min="Min", Max="Max",
                                      Status="Status", Linked="Linked", Expression="Expression", Type="Type")
        s += "\n"
      #  s += TABLE_LENGTH*"_"+"\n"
        for p in self.all.values():
            s += p.__str__() + "\n"
     #   s += TABLE_LENGTH*"="+"\n"
        return s+"\n"
