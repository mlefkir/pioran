"""General class to represent one parameter of a model.

"""
from jax.tree_util import register_pytree_node_class

from .tools import HEADER_PARAMETERS


@register_pytree_node_class
class Parameter():
    """ Class for one parameters, it can be a hyperparameter or model
    parameter.

    The object of this class is then used to create a list of
    parameters with the :class:`~pioran.parameters.ParametersModel` object.

    Parameters
    ----------
    name : :obj:`str`
        Name of the parameter.
    value : :obj:`float`
        Value of the parameter.
    ID : :obj:`int`, optional
        ID of the parameter, default is 1.
    free : :obj:`bool`
        If the parameter is free or fixed.
    hyperparameter : :obj:`bool`, optional
        If the parameter is an hyperparameter of the covariance function or not. The default is True.
    component : :obj:`int`, optional
        Component containing the parameter, default is 1.
    relation : :obj:`Parameter`, optional
        Relation between the parameter and the linked one. The default is None.


    Attributes
    ----------
    name : :obj:`str`
        Name of the parameter.
    value : :obj:`float`
        Value of the parameter.
    free : :obj:`bool`
        If the parameter is free or fixed.
    hyperparameter : :obj:`bool`, optional
        If the parameter is an hyperparameter of the covariance function
        or not. The default is True.
    component : :obj:`int`, optional
        Component containing the parameter, default is 1.
    ID : :obj:`int`, optional
        ID of the parameter, default is 1.
    relation : :obj:`Parameter`, optional
        Relation between the parameter and the linked one. The default is :obj:`None`.
   
   
    Methods
    -------
    tree_flatten()
        Flattens the tree representation of the parameter.
    tree_unflatten(aux_data, children)
        Unflattens the tree representation of the parameter.

    """    
    value: float
    name: str
    free: bool
    ID: int
    hyperparameter: bool
    component: int
    relation: None
    _value: float = None
    _name: str = None

    def __init__(self,name, value, free=True,ID=1,hyperparameter=True,component=1,relation=None):
        self._value = value
        self._name = name
        self.free = free
        self.ID = ID
        self.hyperparameter = hyperparameter
        self.component = component
        self.relation = relation
        

    @property
    def name(self):
        """Get the name of the parameter.
        
        Returns
        -------
        :obj:`str`
            Name of the parameter.
        """
        return self._name
    
    @name.setter
    def name(self,new_name):
        self._name = new_name
        
    @property
    def value(self):
        """Get the value of the parameter.

        Returns
        -------
        :obj:`float`
            Value of the parameter.
        """
        return self._value
    
    @value.setter
    def value(self,new_value):        
        """Set the value of the parameter.

        Will update the value of the parameter in accordance with the relation if the parameter is linked to another one.

        Parameters
        ----------
        value : :obj:`float`
            Value of the parameter.
        """
        self._value = new_value


    def __str__(self) -> str: # pragma: no cover
        """String representation of the parameter.
        
        In the following format:
        
        component  ID  name  value  free  linked  type


        Returns
        -------
        :obj:`str`
            String representation of the parameter.
        """
        
        return HEADER_PARAMETERS.format(Component=self.component if self.component is not None else "N/A",ID=self.ID,Name=self.name,
                                        Value=f"{self.value:5.5e}" if len(
                                            str(self.value)) > 9 else self.value,
                                        Status='Free' if self.free else 'Fixed',
                                        Linked='Yes' if self.relation is not None else 'No',
                                        Type='Hyper-parameter' if self.hyperparameter else 'Model parameter')
    def __repr__(self) -> str: # pragma: no cover
        return self.__str__()
    
    def __repr_html__(self) -> str: # pragma: no cover
        return self.__str__()
    
    
    def tree_flatten(self):
        """Flatten the object for the JAX tree.

        The object is flatten in a tuple containing the dynamic children and the static auxiliary data.
        The dynamic children are the :py:attr:`name` and :py:attr:`value` of the parameter while the static auxiliary data are the attributes
        :py:attr:`free`, :py:attr:`ID`, :py:attr:`hyperparameter`, :py:attr:`component` and :py:attr:`relation`.
        
        Returns
        -------
        :obj:`tuple`
            Tuple containing the children and the auxiliary data.
        """
        
        children = (self.name,self._value)
        aux_data = {'free': self.free,
                    'ID': self.ID,'hyperparameter': self.hyperparameter,
                    'component': self.component,'relation': self.relation}
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Unflatten the object for the JAX tree.

        Parameters
        ----------
        aux_data : :obj:`dict`
            Dictionary containing the static auxiliary data.
        children : :obj:`tuple`
            Tuple containing the dynamic children.

        Returns
        -------
        :obj:`Parameter`
            Parameter object.
        """

        return cls(*children, **aux_data)

