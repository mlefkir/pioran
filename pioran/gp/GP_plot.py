import matplotlib.pyplot as plt
import numpy as np
from .GP_core import GaussianProcess

def plot_prediction(GP: GaussianProcess,figsize,confidenceBands=True,title=None,xlabel=None,ylabel=None,xlim=None,ylim=None):
    """_summary_ : Plot the prediction of the Gaussian Process.

    Parameters
    ----------
    figsize : _type_
        _description_
    confidenceBands : bool, optional
        _description_, by default True
    title : _type_, optional
        _description_, by default None
    xlabel : _type_, optional
        _description_, by default None
    ylabel : _type_, optional
        _description_, by default None
    xlim : _type_, optional
        _description_, by default None
    ylim : _type_, optional
        _description_, by default None
    """  

    import matplotlib.pyplot as plt
    from matplotlib.ticker import AutoMinorLocator

    plt.style.use("https://github.com/mlefkir/beauxgraphs/raw/main/beautifulgraphs.mplstyle")

            
    posterior_mean, posterior_covariance = GP.computePosteriorDistributions()

    fig,ax = plt.subplots(1,1,figsize=figsize)

    ax.errorbar(GP.training_indexes.flatten() , GP.training_observables.flatten(), yerr=GP.training_errors, fmt='.', label='Observation')
    ax.plot(GP.prediction_indexes,posterior_mean, label='GP Prediction')

    ax.set_title(title) if title is not None else None
    ax.set_xlabel(xlabel) if title is not None else None
    ax.set_ylabel(ylabel) if title is not None else None
    ax.set_xlim(xlim) if xlim is not None else None
    ax.set_ylim(ylim) if ylim is not None else None
    
    if confidenceBands:
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
    fig.savefig("test_pythonbase.pdf")
    return fig