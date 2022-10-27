"""Classes for the parameters of covariance functions

"""



class ParametersCovFunction:
    """ Class for the parameters of a covariance function. 

    Initialised with a list of values for the parameters.


    Attributes
    ----------
    names : list of str
        Names of the parameters.
    values : list of float
        Values of the parameters.
    boundaries : dict
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

    def __init__(self, param_values, **kwargs):
        """Constructor method for the ParametersCovFunction class.

        Parameters
        ----------
        param_values : list of float
            Values of the parameters.
        **kwargs : dict
            names : list of str
                Names of the parameters.
            boundaries : list of tuples
                Boundaries of the parameters.
        
        Raises
        ------
        ValueError
            When the number of parameters or boundaries is not the same as the number of names.

        """

        self.values = param_values
        if "names" in kwargs:
            if len(kwargs["names"]) == len(self.values):
                self.names = kwargs["names"]
            else:
                raise ValueError(
                    "The number of names is not the same as the number of parameters.")
        if "boundaries" in kwargs:
            if len(kwargs["boundaries"]) == len(self.values):
                self.boundaries = {}
                for i, b in enumerate(kwargs["boundaries"]):
                    assert len(
                        b) == 2, "The boundaries must be a list of 2 elements."
                    self.boundaries[self.names[i]] = b
            else:
                raise ValueError(
                    "The number of boundaries is not the same as the number of parameters.")

    def update_names(self, names):
        """ Update the parameters names.

        Parameters
        ----------
        names : list of str
            Names of the parameters.
        """
        if len(names) == len(self.values):
            self.names = names
        else:
            raise ValueError(
                "The number of names is not the same as the number of parameters.")

    def update_boundaries(self, boundaries):
        """ Update the boundaries of the parameters

        Parameters
        ----------
        boundaries : list of tuples
        """

        assert len(boundaries) == len(
            self.values), "The number of boundaries is not the same as the number of parameters."
        self.boundaries = {}

        for i, b in enumerate(boundaries):
            assert len(b) == 2, "The boundaries must be a list of 2 elements."
            self.boundaries[self.names[i]] = b

    def add_parameter(self, name, value, boundaries):
        """ Add a parameter to the object.

        Parameters
        ----------
        name : str
            Name of the parameter.
        value : float
            Value of the parameter.
        boundaries : tuple
            Boundaries of the parameter.
        """
        self.names.append(name)
        self.values.append(value)
        self.boundaries[name] = boundaries

    def __getitem__(self, key):
        """ Get the value of a parameter.

        Parameters
        ----------
        key : str
            Name of the parameter.

        Returns
        -------
        value : float
            Value of the parameter with name "key".
        """
        if key in self.names:
            return self.values[self.names.index(key)]
        else:
            raise KeyError(f"Parameter {key} not found.")

    def check_boundaries(self):
        """ Check if the parameters are within the boundaries. 

        Returns
        -------
        bool : True if the parameters are within the boundaries, False otherwise.
        """
        for i in range(len(self.names)):
            if self.values[i] < self.boundaries[self.names[i]][0] or self.values[i] > self.boundaries[self.names[i]][1]:
                return False
        return True

    def print_parameters(self):
        """ Print the parameters. 
        """
        print(f"Parameters of the covariance function: ", *self.names, sep=' ')
        print(f"Values of the parameters: {self.values}")
        print(f"Boundaries of the parameters: {self.boundaries}")
