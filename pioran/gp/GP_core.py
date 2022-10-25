import numpy as np
# modules needed for the GP code a new another time series light curves gaussian processes inference bayesian analysis data variability
from scipy.linalg import cholesky
from numpy.linalg import solve
from .utils import nearest_positive_definite
from .covarfun.covfunction import CovarianceFunction


class GaussianProcess:
    """ Class for the Gaussian Process Regression of 1D data. """


    
    def __init__(self, covariance_function: CovarianceFunction, training_indexes, training_observables, training_errors=None):
        self.covarFunction = covariance_function
        
        # Check if the training arrays have the same shape
        if training_errors is None:
            self.sanity_checks(training_indexes, training_observables)
        else:
            self.sanity_checks(training_indexes, training_observables)
            self.sanity_checks(training_observables, training_errors)

        self.training_indexes = self.reshape_array(training_indexes)
        self.training_observables = self.reshape_array(training_observables)
        self.training_errors = training_errors.flatten() if training_errors is not None else None
        
        ## work in progress 
        ## TODO: Prediction of data
        self.nb_predic_points = 5000
        self.prediction_indexes = self.reshape_array(np.linspace(np.min(self.training_indexes), np.max(self.training_indexes), self.nb_predic_points))

    def reshape_array(self,array):
        """ Reshape the array to a 2D array. """
        return np.reshape(array, (len(array),1)) 

    
    def sanity_checks(self, array_A, array_B):
        """ Check if the lists are of the same shape """
        assert np.shape(array_A)==np.shape(array_B), "The training arrays must have the same shape."
        
    def get_CovarianceMatrix(self, xt, xp,errors=None):
        """ Compute the covariance matrix between two arrays. """
        if  errors is None:
            return self.covarFunction.CovarMatrix(xt, xp)
        
        return self.covarFunction.CovarMatrix(xt, xp)+np.diag(errors**2)

    def get_CovarianceMatrix_training(self):
        Cov_xx = self.get_CovarianceMatrix(self.training_indexes, self.training_indexes,errors = self.training_errors)
        Cov_inv = solve(Cov_xx,np.eye(len(self.training_indexes)))
        alpha = Cov_inv@self.training_observables
        return Cov_xx, Cov_inv, alpha
        
    def computePosteriorDistributions(self):
        """ Compute the posterior distribution for a given query and training set. """
        
        # Compute the covariance matrix between the training indexes
        Cov_xx, Cov_inv, alpha = self.get_CovarianceMatrix_training()
        # Compute the covariance matrix between the training indexes and the prediction indexes
        Cov_xxp = self.get_CovarianceMatrix(self.training_indexes, self.prediction_indexes)
        Cov_xpxp = self.get_CovarianceMatrix(self.prediction_indexes, self.prediction_indexes)
        
        # Compute the posterior mean 
        posterior_mean = Cov_xxp.T@alpha
        
        # Compute the posterior covariance
        posterior_covariance = Cov_xpxp - Cov_xxp.T@Cov_inv@Cov_xxp
        
        return posterior_mean, posterior_covariance

    def computeLogMarginalLikelihood(self):
        """ Compute the log marginal likelihood. """
        Cov_xx, Cov_inv, alpha = self.get_CovarianceMatrix_training()
        # Compute the covariance matrix between the training indexes
        try:
            L = cholesky(Cov_xx)
        except :

            L = cholesky(nearest_positive_definite(Cov_xx))

        return (np.sum(np.log(np.diagonal(L)))+0.5*len(self.training_indexes)*np.log(2*np.pi)+0.5*self.training_observables.T@alpha).flatten()[0]


    def wrapperLogMarginalLikelihood(self, parameters):
        """ Wrapper to compute the log marginal likelihood. """
        self.covarFunction.parameters.values = parameters
        return -self.computeLogMarginalLikelihood()
    