

class ParametersCovFunction:
    """ Class for the parameters of a covariance function. 
    
    attributes:
    - values: list of values of the parameters
    - names: list of names of the parameters
    - boundaries: dictionary of boundaries for the parameters
    
    """
    
    def __init__(self, param_values):
        self.values = param_values
        
    def update_names(self, names):
        """ Update the parameters values. """
        assert len(names) == len(self.values), "The number of names is not the same as the number of parameters."
        self.names = names
        
    def update_boundaries(self, boundaries):
        """ Update the boundaries of the parameters

        Parameters
        ----------
        boundaries : list of tuples
        
        """
        
        assert len(boundaries) == len(self.values), "The number of boundaries is not the same as the number of parameters."
        self.boundaries = {}
        
        for i,b in enumerate(boundaries):
            assert len(b) == 2, "The boundaries must be a list of 2 elements."
            self.boundaries[self.names[i]] = b
        

    def check_boundaries(self):
        """ Check if the parameters are within the boundaries. """
        for i in range(len(self.names)):
            if self.values[i] < self.boundaries[self.names[i]][0] or self.values[i] > self.boundaries[self.names[i]][1]:
                return False
        return True

    def print_parameters(self):
        print(f"Parameters of the covariance function: ",*self.names, sep=' ')
        print(f"Values of the parameters: {self.values}")
        print(f"Boundaries of the parameters: {self.boundaries}")
