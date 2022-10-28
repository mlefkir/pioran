"""Core class for Gaussian process regression.

"""

import numpy as np
from scipy.linalg import cholesky, solve_triangular, solve

from .utils import nearest_positive_definite
from .acvcore import CovarianceFunction
from .parameters import Parameter


class GaussianProcess:
    """ Class for the Gaussian Process Regression of 1D data. 


    Attributes
    ----------
    acvf : CovarianceFunction
        Autocovariance function associated to the Gaussian Process.
    training_indexes : array of shape (n,1)
        Indexes of the training data.
    training_observables : array of shape (n,1)
        Observabled training data.
    training_errors : array of shape (n,1)
        Errors on the training observed data.
    scale_errors : bool
        Scale the errors on the training data by adding a constant, by default True.
    estimate_mean : bool
        Estimate the mean of the training data, by default True.

    Methods
    -------
    get_cov
        Compute the covariance matrix between two arrays.
    compute_predictive_distribution
        Compute the predictive mean and predictive covariance given prediction indexes.
    compute_log_marginal_likelihood
        Compute the log marginal likelihood.
    wrapper_log_marginal_likelihood
        Wrapper to compute the log marginal likelihood.
    wrapper_neg_log_marginal_likelihood
        Wrapper to compute the negative log marginal likelihood.
    """

    def __init__(self, covariance_function: CovarianceFunction, training_indexes, training_observables, training_errors=None, **kwargs):
        """Constructor method for the GaussianProcess class.

        Parameters
        ----------
        covariance_function : CovarianceFunction
            Covariance function associated to the Gaussian Process.
        training_indexes : 1D array
            Indexes of the training data, in this case it is the time.
        training_observables : 1D array
            Observables of the training data, in this it is flux, count-rate or intensity, etc.
        training_errors : 1D array, optional
            Errors on the observables, by default None
        **kwargs : dict
            nb_prediction_points : int, optional
                Number of points to predict, by default 5 * length(training(indexes)).
            prediction_indexes : 1D array, optional
                Indexes of the prediction data, by default np.linspace(np.min(training_indexes),np.max(training_indexes),nb_prediction_points)
            scale_errors : bool, optional
                Scale the errors on the training data by adding a constant, by default True.
            estimate_mean : bool, optional
                Estimate the mean of the training data, by default True.
        """

        self.acvf = covariance_function
        # add a factor to scale the errors
        self.scale_errors = kwargs.get("scale_errors", True)
        if self.scale_errors:
            self.acvf.parameters.append(Parameter(name="nu", 
                                                  value=1.0, 
                                                  bounds=[0.0, 100.0], 
                                                  free=True, 
                                                  hyperpar=False))

        # Check if the training arrays have the same shape
        if training_errors is None:
            self.sanity_checks(training_indexes, training_observables)
        else:
            self.sanity_checks(training_indexes, training_observables)
            self.sanity_checks(training_observables, training_errors)

        # Reshape the arrays
        self.training_indexes = self.reshape_array(training_indexes)
        self.training_observables = self.reshape_array(training_observables)
        self.training_errors = training_errors.flatten() if training_errors is not None else None

        # add the mean of the observed data as a parameter
        self.estimate_mean = kwargs.get("estimate_mean", True)
        if self.estimate_mean:
            self.acvf.parameters.append(Parameter(name="mu", 
                                                  value=np.mean(self.training_observables), 
                                                  bounds=[np.min(self.training_observables), np.max(self.training_observables)],
                                                  free=True, 
                                                  hyperpar=False))
        else:
            print("The mean of the training data is not estimated. Be careful of the data included in the training set.")

        # Prediction of data
        self.nb_predic_points = kwargs.get("nb_prediction_points", 5*len(self.training_indexes))
        self.prediction_indexes = kwargs.get('prediction_indexes', 
                                    self.reshape_array(np.linspace(np.min(self.training_indexes), np.max(self.training_indexes), self.nb_predic_points)))

    def reshape_array(self, array):
        """ Reshape the array to a 2D array with np.shape(array,(len(array),1).

        Parameters
        ----------
        array : 1D array
        
        Returns
        -------
        array : 2D array
            Reshaped array.

        """
        return np.reshape(array, (len(array), 1))

    def sanity_checks(self, array_A, array_B):
        """ Check if the lists are of the same shape 

        Parameters
        ----------
        array_A : array of shape (n,1)
            First array.
        array_B : array  of shape (m,1)
            Second array.
        """
        assert np.shape(array_A) == np.shape(
            array_B), "The training arrays must have the same shape."

    def get_cov(self, xt, xp, errors=None):
        """ Compute the covariance matrix between two arrays. 

        Parameters
        ----------
        xt : array of shape (n,1)
            First array.
        xp : array  of shape (m,1)
            Second array.
        errors : array of shape (n)
            Errors on the observed data, default is None.
            If errors is not None, then the covariance matrix is computed for the training dataset, i.e. with observed 
            data as input (xt=xp=training data) and the errors is the "standard deviation". The total covariance matrix is computed as:

                C = K + nu * sigma^2 * I

                with I the identity matrix, K the covariance matrix, sigma the errors and nu a free parameter in case the errors
                are over or under-estimated.

        Returns
        -------
        array of shape (n,m)
            Covariance matrix between the two arrays.

        """
        # if not errors return the covariance matrix
        if errors is None:
            return self.acvf.get_cov_matrix(xt, xp)
        # if errors and we want to scale them
        if self.scale_errors:
            return self.acvf.get_cov_matrix(xt, xp) + self.acvf.parameters["nu"].value * np.diag(errors**2)
        # if we do not want to scale the errors
        return self.acvf.get_cov_matrix(xt, xp) + np.diag(errors**2)

    def get_cov_training(self):
        """ Compute the covariance matrix and other vectors for the training data.

        Returns
        -------
        Cov_xx : array of shape (n,n)
            Covariance matrix for the training data.
        Cov_inv : array of shape (n,n)
            Inverse of Cov_xx.
        alpha : array of shape (n,1)
            alpha = Cov_inv * training_observables (- mu if mu is estimated)
        """

        Cov_xx = self.get_cov(
            self.training_indexes, self.training_indexes, errors=self.training_errors)
        Cov_inv = solve(Cov_xx, np.eye(len(self.training_indexes)))
        if self.estimate_mean:
            alpha = Cov_inv@(self.training_observables -
                             self.acvf.parameters["mu"].value)
        else:
            alpha = Cov_inv@(self.training_observables)
        return Cov_xx, Cov_inv, alpha

    def compute_predictive_distribution(self, **kwargs):
        """ Compute the predictive mean and the predictive covariance of the GP. 

        The predictive distribution are computed using equations (2.25)  and (2.26) in Rasmussen and Williams (2006)
        Following the notation of the book, x is the training indexes, x* is the predictive indexes, y the training observable, 
        k the covariance function, sig the noise in the observation.
        The predictive distribution is computed as:

        alpha = inv( k(x,x) + sig^2 * I ) * training_observables
        mean = k(x*,x) * alpha
        cov = k(x*,x*) - k(x*,x) * inv( k(x,x) + sig^2 * I ) * k(x,x*)


        Parameters
        ----------
        **kwargs : dict
            prediction_indexes : array of length m, optional
                Indexes of the prediction data, by default np.linspace(np.min(training_indexes),np.max(training_indexes),nb_prediction_points)

        Returns
        -------
        predictive_mean : array of shape (m,1)
            Predictive mean of the GP.
        predictive_cov : array of shape (m,m)
            Predictive covariance of the GP.
        """
        # if we want to change the prediction indexes
        if "prediction_indexes" in kwargs:
            self.prediction_indexes = self.reshape_array(kwargs["prediction_indexes"])

        # Compute the covariance matrix between the training indexes
        _, Cov_inv, alpha = self.get_cov_training()
        # Compute the covariance matrix between the training indexes and the prediction indexes
        Cov_xxp = self.get_cov(self.training_indexes, self.prediction_indexes)
        Cov_xpxp = self.get_cov(self.prediction_indexes,
                                self.prediction_indexes)

        # Compute the predictive mean
        if self.estimate_mean:
            predictive_mean = Cov_xxp.T@alpha + \
                self.acvf.parameters["mu"].value
        else:
            predictive_mean = Cov_xxp.T@alpha
        # Compute the predictive covariance and ensure that the covariance matrix is positive definite
        predictive_covariance = nearest_positive_definite(
            Cov_xpxp - Cov_xxp.T@Cov_inv@Cov_xxp)

        return predictive_mean, predictive_covariance

    def compute_log_marginal_likelihood(self):
        """ Compute the log marginal likelihood of the GP.

        The log marginal likelihood is computed using algorithm (2.1) in Rasmussen and Williams (2006)
        Following the notation of the book, x is the training indexes, x* is the predictive indexes, y the training observable, 
        k the covariance function, sig the noise in the observation.

        Following Simon's notes simply solve of triangular system instead of inverting the matrix:
        
        L = cholesky( k(x,x) + sig^2 * I )
        z = L^-1 * y
        log_marginal_likelihood = - 0.5 * z^T * z - sum(log(diag(L))) - n/2 * log(2*pi)

        Returns
        -------
        log_marginal_likelihood : float
            Log marginal likelihood of the GP.

        """
        Cov_xx = self.get_cov(self.training_indexes, self.training_indexes,
                              errors=self.training_errors)
        # Compute the covariance matrix between the training indexes
        try:
            L = cholesky(Cov_xx, lower=True)
        except:
            L = cholesky(nearest_positive_definite(Cov_xx), lower=True)

        if self.estimate_mean:
            z = solve_triangular(
                L, self.training_observables-self.acvf.parameters["mu"].value, lower=True)
        else:
            z = solve_triangular(L, self.training_observables, lower=True)

        return -((np.sum(np.log(np.diagonal(L))) + 0.5 * len(self.training_indexes) * np.log(2*np.pi) + 0.5 * (z.T@z)).flatten()[0])

    def wrapper_log_marginal_likelihood(self, parameters):
        """ Wrapper to compute the log marginal likelihood in function of the (hyper)parameters. 

        Parameters
        ----------
        parameters : array of shape (n)
            (Hyper)parameters of the covariance function.
            
        Returns
        -------
        float 
            Log marginal likelihood of the GP.
        """
        self.acvf.parameters.values = parameters
        return self.compute_log_marginal_likelihood()

    def wrapper_neg_log_marginal_likelihood(self, parameters):
        """ Wrapper to compute the negative log marginal likelihood in function of the (hyper)parameters. 

        Parameters
        ----------
        parameters : array of shape (n)
            (Hyper)parameters of the covariance function.
            
        Returns
        -------
        float
            Negative log marginal likelihood of the GP.
        """
        self.acvf.parameters.values = parameters
        return -self.compute_log_marginal_likelihood()

    def __str__(self) -> str:
        """String representation of the GP object.
        
        Returns
        -------
        str
            String representation of the GP object.        
        """
        s = 31*"=" +" Gaussian Process "+31*"="+"\n\n"
        s += f"Marginal log likelihood : {self.compute_log_marginal_likelihood():.5f}\n"
        s += self.acvf.__str__()
        return s
