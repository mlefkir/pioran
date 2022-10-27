"""Collection of functions for plotting the results of the Gaussian Process Regression.

"""

import numpy as np
import matplotlib.pyplot as plt    
from matplotlib.ticker import AutoMinorLocator

from .core import GaussianProcess


plt.style.use("https://github.com/mlefkir/beauxgraphs/raw/main/beautifulgraphs.mplstyle")


def plot_prediction(GP: GaussianProcess,filename,figsize,confidence_bands=True,title=None,xlabel=None,ylabel=None,xlim=None,ylim=None):
    """Plot the prediction of the Gaussian Process.

    Parameters
    ----------
    GP : GaussianProcess
        Gaussian Process object.
    filename : str
        Name of the file to save the figure.
    figsize : tuple
        Size of the figure.
    confidence_bands : bool, optional
        Plot the confidence bands, by default True
    title : str, optional
        Title of the plot, by default None
    xlabel : str, optional
        Label for the x-axis, by default None
    ylabel : str, optional
        Label for the y-axis, by default None
    xlim : tuple of floats, optional
        Limits of the x-axis, by default None
    ylim : tuple of floats, optional
        Limits of the y-axis, by default None
    """  
    fig,ax = plt.subplots(1,1,figsize=figsize)
     
    # get predictions from GP
    posterior_mean, posterior_covariance = GP.computePosteriorDistributions()
    
    ax.errorbar(GP.training_indexes.flatten() , GP.training_observables.flatten(), yerr=GP.training_errors, fmt='.', label='Observation')
    ax.plot(GP.prediction_indexes,posterior_mean, label='GP Prediction')

    ax.set_title(title) if title is not None else None
    ax.set_xlabel(xlabel) if title is not None else None
    ax.set_ylabel(ylabel) if title is not None else None
    ax.set_xlim(xlim) if xlim is not None else None
    ax.set_ylim(ylim) if ylim is not None else None
    
    if confidence_bands:
        std = np.sqrt(np.diag(posterior_covariance))
        hi = (posterior_mean.T-std).flatten()
        lo = (posterior_mean.T+std).flatten()
        hi2 = (posterior_mean.T-2*std).flatten()
        lo2 = (posterior_mean.T+2*std).flatten()
        ax.fill_between(GP.prediction_indexes.T.flatten(),(hi),(lo),alpha=0.1,label=r"$1\sigma$")
        ax.fill_between(GP.prediction_indexes.T.flatten(),(hi2),(lo2),alpha=0.1,label=r"$2\sigma$")
    
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.legend()
    
    fig.tight_layout()
    fig.savefig(f"{filename}.pdf")