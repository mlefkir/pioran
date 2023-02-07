"""General classes for operations on one parameter

"""


from .tools import TYPE_NUMBER, HEADER_PARAMETERS


class Parameter():
    """Class for one parameters, it can be a hyperparameter or a mean
    parameter.

    The object of this class is then used to create a list of
    parameters with the :obj:`ParametersModel` object.

    Parameters
    ----------
    name : :obj:`str`
        Name of the parameter.
    value : :obj:`float`
        Value of the parameter.
    bounds : :obj:`list`
        Bounds of the parameter.
    ID : :obj:`int`, optional
        ID of the parameter, default is 1.
    free : :obj:`bool`
        If the parameter is free or fixed.
    hyperpar : :obj:`bool`, optional
        If the parameter is an hyperparameter of the covariance function or not. The default is True.
    linked : :obj:`bool`, optional
        If the parameter is linked to another one. The default is False.
    relation : :obj:`Parameter`, optional
        Relation between the parameter and the linked one. The default is None.


    Attributes
    ----------
    name : :obj:`str`
        Name of the parameter.
    value : :obj:`float`
        Value of the parameter.
    bounds : :obj:`list`
        Bounds of the parameter. Should be a list of two elements, the
        first one is the lower bound and the second one is the upper bound.
    free : :obj:`bool`
        If the parameter is free or fixed.
    hyper : :obj:`bool`, optional
            If the parameter is an hyperparameter of the covariance function
            or not. The default is True.
    ID : :obj:`int`, optional
        ID of the parameter, default is 1.
    linked : :obj:`bool`, optional
        If the parameter is linked to another one. The default is False.
    relation : :obj:`Parameter`, optional
        Relation between the parameter and the linked one. The default is None.
    expression : :obj:`str`, optional
        Expression of the parameter. The default is ''.
    fullname : :obj:`str`, optional
        Full name of the parameter. The default is f"{name}[{ID}]"

    """

    def __init__(self, name: str, value: float, bounds: list = [None, None], ID: int = 1, free: bool = True, hyperpar=True, linked=False, relation=None):
        """Constructor method for the Parameter class.
        """
        
        self.linked = linked
        self.name = name
        self.value = value
        self.bounds = bounds
        self.free = free
        self.hyperpar = hyperpar
        self.ID = ID
        self.relation = relation
        self.expression = ''
        self.fullname = f"{name}[{ID}]"
        self.relation = None

    def __eq__(self, other) -> bool:
        """Compare two parameters.

        Parameters
        ----------
        other : Parameter
            Parameter to compare.

        Returns
        -------
        bool
            True if the parameters are the same, False otherwise.

        """
        return self.name == other.name and self.value == other.value and self.bounds == other.bounds and self.free == other.free and self.hyperpar == other.hyperpar

    @property
    def relation(self):
        """Get the relation of the parameter.

        Returns
        -------
        Parameter or Operations
            Relation of the parameter.
        """
        #  self.relation = self._relation
        return self._relation

    @relation.setter
    def relation(self, relation):
        """Set the relation of the parameter.

        Parameters
        ----------
        relation : Parameter or Operations
            Relation of the parameter.
        """
        if relation is not None:
            self._relation = relation
            self.linked = True
            self._value = relation.value
            self.expression = relation.expression

    @property
    def value(self) -> TYPE_NUMBER:
        """Get the value of the parameter.

        Returns
        -------
        float
            Value of the parameter.
        """

        # this is to update the value of the parameter if it is linked to another parameter
        self.value = self._value
        return self._value

    @value.setter
    def value(self, value):
        """Set the value of the parameter.

        Will update the value of the parameter in accordance with the relation if the parameter is linked to another one.

        Parameters
        ----------
        value : float
            Value of the parameter.
        """

        if self.linked:
            self._value = self.relation.value
        else:
            self._value = value

    def __add__(self, other) -> 'SumParameters':
        """Add two parameters.

        Overload of the + operator for the Parameter class.

        Parameters
        ----------
        other : Parameter or TYPE_NUMBER
            Parameter or number to add.

        Returns
        -------
        SumParameters
            Sum of the two parameters.

        Raises
        ------
        TypeError
            If the other parameter is not a Parameter or a number.
        """
        if isinstance(other, Parameter):
            return SumParameters(self, other)
        elif isinstance(other, TYPE_NUMBER):
            if other == 0:
                return SumParameters(self, Parameter(name='constant', value=0, bounds=[None, None], free=False, hyperpar=False), is_scalar=True)
            return SumParameters(self, Parameter(name='constant', value=other, bounds=[None, None], free=False, hyperpar=False), is_scalar=True)
        else:
            raise TypeError(f"Cannot add a parameter and a {type(other)}")

    def __neg__(self) -> 'ProductParameters':
        """Negate a parameter.

        Overload of the negate operator for the Parameter class.

        Returns
        -------
        ProductParameters
            Negated parameter.
        """
        return ProductParameters(self, Parameter(name='constant', value=-1, bounds=[None, None], free=False, hyperpar=False), is_scalar=True)

    def __sub__(self, other) -> 'SubtractionParameters':
        """Subtract two parameters.

        Overload of the - operator for the Parameter class.

        Parameters
        ----------
        other : Parameter or TYPE_NUMBER
            Parameter or number to subtract.

        Returns
        -------
        SubtractionParameters
            Sum of the two parameters.

        Raises
        ------
        TypeError
            If the other parameter is not a Parameter or a number.
        """
        if isinstance(other, Parameter):
            return SubtractionParameters(self, other)
        elif isinstance(other, TYPE_NUMBER):
            return SubtractionParameters(self, Parameter(name='constant', value=other, bounds=[None, None], free=False, hyperpar=False), is_scalar=True)
        else:
            raise TypeError(f"Cannot subtract a parameter and a {type(other)}")

    def __mul__(self, other) -> 'ProductParameters':
        """Multiply two parameters.

        Overload of the * operator for the Parameter class.

        Parameters
        ----------
        other : Parameter or TYPE_NUMBER
            Parameter or number to multiply.

        Returns
        -------
        ProductParameters
            Product of the two parameters.

        Raises
        ------
        TypeError
            If the other parameter is not a Parameter or a number.
        """
        if isinstance(other, Parameter):
            return ProductParameters(self, other)
        elif isinstance(other, TYPE_NUMBER):
            return ProductParameters(self, Parameter(name='constant', value=other, bounds=[None, None], free=False, hyperpar=False), is_scalar=True)
        else:
            raise TypeError(f"Cannot multiply a parameter and a {type(other)}")

    def __pow__(self, other) -> 'PowParameters':
        """Exponentiation of two parameters.

        Overload of the ** operator for the Parameter class.

        Parameters
        ----------
        other : Parameter or TYPE_NUMBER
            Parameter or number to use as the exponent.

        Returns
        -------
        PowerParameters
            Exponentiation of parameters.

        Raises
        ------
        TypeError
            If the other parameter is not a Parameter or a number.
        """
        if isinstance(other, Parameter):
            return PowParameters(self, other)
        elif isinstance(other, TYPE_NUMBER):
            return PowParameters(self, Parameter(name='constant', value=other, 
            bounds=[None, None], free=False, hyperpar=False), is_scalar=True)
        else:
            raise TypeError(
                f"Cannot exponentiate a parameter and a {type(other)}")

    def __truediv__(self, other) -> 'DivisionParameters':
        """Divide two parameters.

        Overload of the / operator for the Parameter class.

        Parameters
        ----------
        other : Parameter or TYPE_NUMBER
            Parameter or number to divide by.

        Returns
        -------
        DivisionParameters
            Division of the two parameters.

        Raises
        ------
        ZeroDivisionError
            If the other parameter is 0 or a Parameter whose value is 0.
        TypeError
            If the other parameter is not a Parameter or a number.
        """
        if isinstance(other, Parameter):
            if other.value == 0:
                raise ZeroDivisionError(
                    "Cannot divide by a parameter with a zero value")
            return DivisionParameters(self, other)
        elif isinstance(other, TYPE_NUMBER):
            if other == 0:
                raise ZeroDivisionError('Cannot divide by zero')
            return DivisionParameters(self, Parameter(name='constant', value=other, bounds=[None, None], free=False, hyperpar=False), is_scalar=True)
        else:
            raise TypeError(f"Cannot divide a parameter and a {type(other)}")

    def __str__(self) -> str:
        """String representation of the Parameter class.

        Prints the name, value, bounds, status(free or fixed), linked (yes or no), expression (if linked) and hyperpar attributes of the parameter.

        Returns
        -------
        str
            String representation of the Parameter class.
            In the form of {name value min max statut type}
        """
        bnd_str = []
        for bnd in self.bounds:
            if bnd is not None:
                if len(str(bnd)) > 6:
                    bnd_str.append(f"{bnd:5.3e}")
                else:
                    bnd_str.append(f"{bnd}")
            else:
                bnd_str.append("None")

        self.bounds[0] if self.bounds[0] is not None else "None",
        return HEADER_PARAMETERS.format(ID=self.ID,Name=self.name,
                                        Value=f"{self.value:5.5e}" if len(
                                            str(self.value)) > 9 else self.value,
                                        Min=bnd_str[0],
                                        Max=bnd_str[1],
                                        Status='Free' if self.free else 'Fixed',
                                        Linked='Yes' if self.linked else 'No',
                                        Expression=self.expression,
                                        Type='Hyper-parameter' if self.hyperpar else 'Model parameter')
        
    __radd__ = __add__
    __rmul__ = __mul__


class Operations(Parameter):
    """Base class for operations on parameters.

    Will be used to define the operations between parameters such as
    sum, product, exponentiation, subtraction and division.    
    """

    @property
    def expression(self):
        return self._expression

    @expression.setter
    def expression(self, expression):
        self._expression = expression

    @property
    def fullname(self) -> str:
        self._fullname = self.expression
        return self._fullname

    @fullname.setter
    def fullname(self, fullname):
        self._fullname = fullname


class SumParameters(Operations, Parameter):
    """Class for the sum of parameters.

    Represents the mathematical operation: first+second.

    Attributes
    ----------
    first : :obj:`Parameter`
        First parameter.
    second : :obj:`Parameter`
        Second parameter
    is_scalar : :obj:`bool`
        True if the second parameter is a scalar, False otherwise.
    fullname : :obj:`str`
        Full name of the resulting parameter.
        
    """

    def __init__(self, first: Parameter, second: Parameter, is_scalar=False):
        """Constructor of the class.

        Parameters
        ----------
        first : :obj:`Parameter`
            First parameter.
        second : :obj:`Parameter`
            Second parameter.
        is_scalar : :obj:`bool`
            True if the second parameter is a scalar, False otherwise.
            
        """

        self.first = first
        self.second = second
        self.value = self.first.value + self.second.value

        if is_scalar:
            self.expression = f"{self.first.fullname} + {self.second.value}"
        else:
            self.expression = f"{self.first.fullname} + {self.second.fullname}"
        self.fullname = self.expression

    @property
    def value(self) -> TYPE_NUMBER:
        """Get the value of the parameter from the product of the two parameters values.

        Returns
        -------
        Value of the parameter.
        """

        self._value = self.first.value + self.second.value
        return self._value

    @value.setter
    def value(self, value):
        self._value = value


class SubtractionParameters(Operations, Parameter):
    """Class for the subtraction of parameters.
    
    Represents the mathematical operation: first-second.

    Attributes
    ----------
    first : :obj:`Parameter`
        First parameter.
    second : :obj:`Parameter`
        Second parameter.
    is_scalar : :obj:`bool`
        True if the second parameter is a scalar, False otherwise.
    fullname : :obj:`str`
        Full name of the resulting parameter.

    """

    def __init__(self, first: Parameter, second: Parameter, is_scalar=False):
        """Constructor of the class.

        Parameters
        ----------
        first : :obj:`Parameter`
            First parameter.
        second : :obj:`Parameter`
            Second parameter.
        is_scalar : :obj:`bool`
            True if the second parameter is a scalar, False otherwise.
        """

        self.first = first
        self.second = second
        self.value = self.first.value - self.second.value

        if is_scalar:
            self.expression = f"{self.first.fullname} - {self.second.value}"
        else:
            self.expression = f"{self.first.fullname} - {self.second.fullname}"
        self.fullname = self.expression

    @property
    def value(self) -> TYPE_NUMBER:
        """Get the value of the parameter from the product of the two parameters values.

        Returns
        -------
        float
            Value of the parameter.
        """

        self._value = self.first.value - self.second.value
        return self._value

    @value.setter
    def value(self, value):
        self._value = value


class PowParameters(Operations, Parameter):
    """Class for the exponentiation of parameters.

    Represents the mathematical operation: first^second.
    
    Attributes
    ----------
    first : :obj:`Parameter`
        First parameter to exponentiate.
    second : :obj:`Parameter`
        Second parameter, is the exponent.
    is_scalar : :obj:`bool`
        True if the second parameter is a scalar, False otherwise.
    fullname : :obj:`str`
        Full name of the resulting parameter.

    """

    def __init__(self, first: Parameter, second: Parameter, is_scalar=False):
        """Constructor of the class.

        Parameters
        ----------
        first : :obj:`Parameter`
            First parameter.
        second : :obj:`Parameter`
            Second parameter.
        is_scalar : :obj:`bool`
            True if the second parameter is a scalar, False otherwise.
        """

        self.first = first
        self.second = second
        self.value = self.first.value ** self.second.value

        if is_scalar:
            firstterm = f"({self.first.fullname})" if isinstance(self.first, SumParameters |
                                                                 SubtractionParameters | DivisionParameters | ProductParameters) else self.first.fullname
            self.expression = f"{firstterm} ** {self.second.value}"
        else:
            firstterm = f"({self.first.fullname})" if isinstance(self.first, SumParameters |
                                                                 SubtractionParameters | DivisionParameters | ProductParameters) else self.first.fullname
            secondterm = f"({self.second.fullname})" if isinstance(self.second, SumParameters |
                                                                   SubtractionParameters | DivisionParameters | ProductParameters) else self.second.fullname

            self.expression = f"{firstterm} ** {secondterm}"
        self.fullname = self.expression

    @property
    def value(self) -> TYPE_NUMBER:
        """Get the value of the parameter from the exponentiation of the two parameters values.

        Returns
        -------
        float
            Value of the parameter.
        """

        self._value = self.first.value ** self.second.value
        return self._value

    @value.setter
    def value(self, value):
        self._value = value


class DivisionParameters(Operations, Parameter):
    """Class for the division of parameters.
    
    Represents the mathematical operation: first/second.

    Attributes
    ----------
    first : :obj:`Parameter`
        First parameter.
    second : :obj:`Parameter`
        Second parameter.
    is_scalar : :obj:`bool`
        True if the second parameter is a scalar, False otherwise.
    fullname : :obj:`str`
        Full name of the resulting parameter.

    """

    def __init__(self, first: Parameter, second: Parameter, is_scalar=False):
        """Constructor of the class.

        Parameters
        ----------
        first : :obj:`Parameter`
            First parameter.
        second : :obj:`Parameter`
            Second parameter.
        is_scalar : :obj:`bool`
            True if the second parameter is a scalar, False otherwise.
        """

        self.first = first
        self.second = second
        self.value = self.first.value / self.second.value

        if is_scalar:
            self.expression = f"{self.first.fullname} / {self.second.value}"

        else:
            firstterm = f"({self.first.fullname})" if isinstance(
                self.first, SumParameters | SubtractionParameters) else self.first.fullname
            secondterm = f"({self.second.fullname})" if isinstance(
                self.second, SumParameters | SubtractionParameters) else self.second.fullname

            self.expression = f"{firstterm} / {secondterm}"
        self.fullname = self.expression

    @property
    def value(self) -> TYPE_NUMBER:
        """Get the value of the parameter from the division of the two parameters values.

        Returns
        -------
        TYPE_NUMBER
            Value of the division.
        """

        self._value = self.first.value / self.second.value
        return self._value

    @value.setter
    def value(self, value):
        self._value = value


class ProductParameters(Operations, Parameter):
    """Class for the product of parameters.
    
    Represents the mathematical operation: first*second.

    Attributes
    ----------
    first : :obj:`Parameter`
        First parameter.
    second : :obj:`Parameter`
        Second parameter.
    is_scalar : :obj:`bool`
        True if the second parameter is a scalar, False otherwise.
    fullname : :obj:`str`
        Full name of the resulting parameter.

    """

    def __init__(self, first: Parameter, second: Parameter, is_scalar=False):
        """Constructor of the class.
        
        Parameters
        ----------
        first : :obj:`Parameter`
            First parameter.
        second : :obj:`Parameter`
            Second parameter.
        is_scalar : :obj:`bool`
            True if the second parameter is a scalar, False otherwise.
        """

        self.first = first
        self.second = second
        self.value = self.first.value * self.second.value
        if is_scalar:
            if self.second.value == -1:
                self.expression = f"-{self.first.fullname}"
            else:
                self.expression = f"{self.first.fullname} * {self.second.value}"
        else:
            firstterm = f"({self.first.fullname})" if isinstance(
                self.first, SumParameters | SubtractionParameters | PowParameters) else self.first.fullname
            secondterm = f"({self.second.fullname})" if isinstance(
                self.second, SumParameters | SubtractionParameters | PowParameters) else self.second.fullname

            self.expression = f"{firstterm} * {secondterm}"
        self.fullname = self.expression

    @property
    def value(self) -> TYPE_NUMBER:
        """Get the value of the parameter from the product of the two parameters values.

        Returns
        -------
        :obj:`float`
            Value of the product.
        """

        self._value = self.first.value * self.second.value
        return self._value

    @value.setter
    def value(self, value):
        self._value = value
