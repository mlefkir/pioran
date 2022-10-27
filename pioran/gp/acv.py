"""Collection of classes for covariance functions

- Exponential
- Squared Exponential
- Matern 3/2
- Matern 5/2
- Rational Quadratic

"""

import numpy as np
from .acvcore import CovarianceFunction, EuclideanDistance


class Exponential(CovarianceFunction):
    """ Class for the exponential covariance function.

    K(r) = variance * exp( - r / lengthscale)

    Attributes
    ----------
    parameters : ParametersCovFunction
        Parameters of the covariance function.

    Methods
    -------
    get_cov_matrix
        Compute the covariance matrix between two arrays.
    """

    def __init__(self, parameters_values, **kwargs):
        """Constructor of the covariance function inherited from the CovarianceFunction class.

        Parameters
        ----------
        parameters_values : list of float or ParametersCovFunction
            Values of the parameters of the covariance function.

        Raises
        ------
        TypeError
            If the parameters_values is not a list of float or a ParametersCovFunction.

        """
        # initialise the parameters and check
        CovarianceFunction.__init__(self, parameters_values, names=[
                                    'variance', 'lengthscale'], boundaries=[[0, np.inf], [0, np.inf]])

    def get_cov_matrix(self, xq, xp):
        """Compute the covariance matrix between two arrays for the exponential covariance function.

        K(xq,xp) = variance * exp( - (xq-xp) / lengthscale )

        The term (xq-xp) is computed using the Euclidean distance from the module covarfun.distance

        Parameters
        ----------
        xq : array of shape (n,1)
            First array.
        xp : array  of shape (m,1)
            Second array.

        Returns
        -------
        K : array of shape (n,m)
            Covariance matrix.
        """
        # Compute the Euclidean distance between the query and the points
        dist = EuclideanDistance(xq, xp)
        # Compute the covariance matrix
        covMat = self.parameters['variance'] * \
            np.exp(- dist / self.parameters['lengthscale'])

        return covMat


class SquareExponential(CovarianceFunction):
    """ Class for the squared exponential covariance function.

    K(r) = variance * exp( -1/2 * r^2 / lengthscale^2)

    Attributes
    ----------
    parameters : ParametersCovFunction
        Parameters of the covariance function.

    Methods
    -------
    get_cov_matrix
        Compute the covariance matrix between two arrays.
    """

    def __init__(self, parameters_values, **kwargs):
        """Constructor of the covariance function inherited from the CovarianceFunction class.

        Parameters
        ----------
        parameters_values : list of float or ParametersCovFunction
            Values of the parameters of the covariance function.

        Raises
        ------
        TypeError
            If the parameters_values is not a list of float or a ParametersCovFunction.

        """
        # initialise the parameters
        CovarianceFunction.__init__(self, parameters_values, names=[
                                    'variance', 'lengthscale'], boundaries=[[0, np.inf], [0, np.inf]])

    def get_cov_matrix(self, xq, xp):
        """Compute the covariance matrix between two arrays for the square exponential covariance function.

        K(xq,xp) = variance * exp( -1/2 * (xq-xp)^2 / lengthscale^2)

        The term (xq-xp) is computed using the Euclidean distance from the module covarfun.distance

        Parameters
        ----------
        xq : array of shape (n,1)
            First array.
        xp : array  of shape (m,1)
            Second array.

        Returns
        -------
        K : array of shape (n,m)
            Covariance matrix.
        """
        # Compute the Euclidean distance between the query and the points
        dist = EuclideanDistance(xq, xp)
        # Compute the covariance matrix
        covMat = self.parameters['variance'] * \
            np.exp(-0.5 * dist**2 / self.parameters['lengthscale']**2)
        return covMat


class Matern32(CovarianceFunction):
    """ Class for the Matern 3/2 covariance function.

    K(r) = variance * (1 + sqrt(3) * r / lengthscale) * exp( -sqrt(3) * r / lengthscale)

    Attributes
    ----------
    parameters : ParametersCovFunction
        Parameters of the covariance function.

    Methods
    -------
    get_cov_matrix
        Compute the covariance matrix between two arrays.
    """

    def __init__(self, parameters_values, **kwargs):
        """Constructor of the covariance function inherited from the CovarianceFunction class.

        Parameters
        ----------
        parameters_values : list of float or ParametersCovFunction
            Values of the parameters of the covariance function.

        Raises
        ------
        TypeError
            If the parameters_values is not a list of float or a ParametersCovFunction.

        """
        # initialise the parameters
        CovarianceFunction.__init__(self, parameters_values, names=[
                                    'variance', 'lengthscale'], boundaries=[[0, np.inf], [0, np.inf]])

    def get_cov_matrix(self, xq, xp):
        """Compute the covariance matrix between two arrays for the Matern 3/2 covariance function.

        K(xq,xp) = variance *  (1 + sqrt(3) * (xq-xp) / lengthscale) * exp( -sqrt(3) * (xq-xp)/ lengthscale)

        The term (xq-xp) is computed using the Euclidean distance.

        Parameters
        ----------
        xq : array of shape (n,1)
            First array.
        xp : array  of shape (m,1)
            Second array.

        Returns
        -------
        K : array of shape (n,m)
            Covariance matrix.
        """
        # Compute the Euclidean distance between the query and the points
        dist = EuclideanDistance(xq, xp)
        # Compute the covariance matrix
        covMat = self.parameters['variance'] * (1 + np.sqrt(3) * dist / self.parameters['lengthscale']) * \
            np.exp(- np.sqrt(3) * dist / self.parameters['lengthscale'])
        return covMat


class Matern52(CovarianceFunction):
    """ Class for the Matern 5/2 covariance function.

    K(r) = variance * (1 + sqrt(5) * r / lengthscale + 5 * r^2 / (3 * lengthscale^2) ) * exp( -sqrt(5) * r / lengthscale)

    Attributes
    ----------
    parameters : ParametersCovFunction
        Parameters of the covariance function.

    Methods
    -------
    get_cov_matrix
        Compute the covariance matrix between two arrays.
    """

    def __init__(self, parameters_values, **kwargs):
        """Constructor of the covariance function inherited from the CovarianceFunction class.

        Parameters
        ----------
        parameters_values : list of float or ParametersCovFunction
            Values of the parameters of the covariance function.

        Raises
        ------
        TypeError
            If the parameters_values is not a list of float or a ParametersCovFunction.

        """
        # initialise the parameters
        CovarianceFunction.__init__(self, parameters_values, names=[
                                    'variance', 'lengthscale'], boundaries=[[0, np.inf], [0, np.inf]])

    def get_cov_matrix(self, xq, xp):
        """Compute the covariance matrix between two arrays for the Matern 5/2 covariance function.

        K(xq,xp) = variance *  (1 + sqrt(5) * (xq-xp) / lengthscale  + 5 * (xq-xp)^2 / (3 * lengthscale^2) ) * exp( -sqrt(5) * (xq-xp)/ lengthscale)

        The term (xq-xp) is computed using the Euclidean distance.

        Parameters
        ----------
        xq : array of shape (n,1)
            First array.
        xp : array  of shape (m,1)
            Second array.

        Returns
        -------
        K : array of shape (n,m)
            Covariance matrix.
        """
        # Compute the Euclidean distance between the query and the points
        dist = EuclideanDistance(xq, xp)
        # Compute the covariance matrix
        covMat = self.parameters['variance'] * (1 + np.sqrt(5) * dist / self.parameters['lengthscale'] + 5 * dist**2 / (
            3 * self.parameters['lengthscale'])) * np.exp(- np.sqrt(3) * dist / self.parameters['lengthscale'])
        return covMat


class RationalQuadratic(CovarianceFunction):
    """ Class for the rational quadratic covariance function.

    K(r) = variance * (1 + r^2 / (2 * alpha * lengthscale^2) )^(-alpha)

    with: alpha, lengthscale > 0

    Attributes
    ----------
    parameters : ParametersCovFunction
        Parameters of the covariance function.

    Methods
    -------
    get_cov_matrix
        Compute the covariance matrix between two arrays.
    """

    def __init__(self, parameters_values, **kwargs):
        """Constructor of the covariance function inherited from the CovarianceFunction class.

        Parameters
        ----------
        parameters_values : list of float or ParametersCovFunction
            Values of the parameters of the covariance function.

        Raises
        ------
        TypeError
            If the parameters_values is not a list of float or a ParametersCovFunction.

        """
        # initialise the parameters
        CovarianceFunction.__init__(self, parameters_values, names=[
                                    'variance', 'alpha', 'lengthscale'], boundaries=[[0, np.inf], [0, np.inf], [0, np.inf]])

    def get_cov_matrix(self, xq, xp):
        """Compute the covariance matrix between two arrays for the rational quadratic covariance function.

        K(xq,xp) = variance *  (1 + (xq-xp)^2 / (2 * alpha * lengthscale^2) ) ^(-alpha)

        The term (xq-xp) is computed using the Euclidean distance.

        Parameters
        ----------
        xq : array of shape (n,1)
            First array.
        xp : array  of shape (m,1)
            Second array.

        Returns
        -------
        K : array of shape (n,m)
            Covariance matrix.
        """
        # Compute the Euclidean distance between the query and the points
        dist = EuclideanDistance(xq, xp)
        # Compute the covariance matrix
        covMat = self.parameters['variance'] * (1 + dist**2 / (
            2 * self.parameters['alpha'] * self.parameters['lengthscale']**2) ** (- self.parameter['alpha']))
        return covMat
