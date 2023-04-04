"""Collection of functions for plotting the results of the Gaussian Process Regression.

"""

import numpy as np
import matplotlib.pyplot as plt    
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import scipy 

import plotly.graph_objects as go

from .core import GaussianProcess




# plt.style.use("https://github.com/mlefkir/beauxgraphs/raw/main/beautifulgraphs.mplstyle")

def plot_prediction_plotly(gp, name, figsize=(18, 5), xlabel="Time", ylabel="Flux",title="Light Curve",show=False):
    predict_mean, predict_var = gp.compute_predictive_distribution()

    std = np.sqrt(np.diag(predict_var))
    fig = go.Figure()
    if gp.training_errors is not None:
        fig.add_trace(go.Scatter(
            x=gp.training_indexes.flatten(), 
            y=gp.training_observables.flatten(),
            mode='markers',
            name='observations',
            error_y=dict(
                array=gp.training_errors.flatten(),
                symmetric=True,
                thickness=1.5,
                width=3,
            ),
            marker=dict( size=8)
        ))
    else:
        fig.add_trace(go.Scatter(
            x=gp.training_indexes.flatten(), 
            y=gp.training_observables.flatten(),
            mode='markers',
            name='observations',
            marker=dict( size=8)
        ))
    fig.add_trace(go.Scatter(
        x=gp.prediction_indexes.flatten(), y=predict_mean.flatten(),
        name='GP'
    ))
    fig.add_trace(go.Scatter(
            name = r'$\pm 1\sigma$',
            x = gp.prediction_indexes.flatten(),
            y = (predict_mean.T+std).flatten(),
            mode='lines',
            marker=dict(color="#444"),
            line=dict(width=0),
            showlegend=True
        ))
    fig.add_trace(go.Scatter(
            name='Lower Bound',
            x=gp.prediction_indexes.flatten(),
            y=(predict_mean.T-std).flatten(),
            marker=dict(color="#444"),
            line=dict(width=0),
            mode='lines',
            fillcolor='rgba(68, 68, 68, 0.3)',
            fill='tonexty',
            showlegend=False
    ))
    fig.add_trace(go.Scatter(
            name = r'$\pm 2\sigma$',
            x = gp.prediction_indexes.flatten(),
            y = (predict_mean.T+2*std).flatten(),
            mode='lines',
            marker=dict(color="#444"),
            line=dict(width=0),
            showlegend=True
        ))
    fig.add_trace(go.Scatter(
            name='Lower Bound',
            x=gp.prediction_indexes.flatten(),
            y=(predict_mean.T-2*std).flatten(),
            marker=dict(color="#444"),
            line=dict(width=0),
            mode='lines',
            fillcolor='rgba(34, 23, 223, 0.3)',
            fill='tonexty',
            showlegend=False
    ))
    fig.update_layout(
        yaxis_title='Flux',
        xaxis_title='Time',
        title=title,
        hovermode="x"
    )
    fig.write_html(f"{name}.html")
    if show:
        fig.show()


def plot_prediction(GP: GaussianProcess,filename,figsize=(16,6),confidence_bands=True,title=None,xlabel=None,ylabel=None,xlim=None,ylim=None):
    """Plot the prediction of the Gaussian Process.

    Parameters
    ----------
    GP: GaussianProcess
        Gaussian Process object.
    filename: str
        Name of the file to save the figure.
    figsize: tuple
        Size of the figure.
    confidence_bands: bool, optional
        Plot the confidence bands, by default True
    title: str, optional
        Title of the plot, by default None
    xlabel: str, optional
        Label for the x-axis, by default None
    ylabel: str, optional
        Label for the y-axis, by default None
    xlim: tuple of floats, optional
        Limits of the x-axis, by default None
    ylim: tuple of floats, optional
        Limits of the y-axis, by default None
    """  
    fig,ax = plt.subplots(1,1,figsize=figsize)
     
    # get predictions from GP
    posterior_mean, posterior_covariance = GP.compute_predictive_distribution()
    
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
    fig.savefig(f"{filename}_regression.pdf")
    fig.show()
    

def plot_residuals(GP: GaussianProcess,filename,figsize=(10,10),maxlag=None,title=None):
    """Plot the residuals of the Gaussian Process inference


    Parameters
    ----------
    GP: GaussianProcess
        Gaussian Process object.
    filename: str
        Name of the file to save the figure.
    figsize: tuple
        Size of the figure.
    maxlag: int, optional
        Maximum lag to plot, by default None
    title: str, optional
        Title of the plot, by default None
    """
    
    if maxlag is None:
        maxlag = len(GP.training_indexes)-2
    else:
        maxlag = np.rint(maxlag)

    predict_mean_train, _ = GP.compute_predictive_distribution(prediction_indexes=GP.training_indexes)
  
    residuals = GP.training_observables.flatten() - predict_mean_train.flatten()
    scaled_residuals = (residuals/GP.training_errors).flatten()
    max_scale_res = np.rint(np.max(scaled_residuals))

    fig = plt.figure(tight_layout=True,figsize=figsize)

    gs0 = fig.add_gridspec(2, 1)
    gs00 = gs0[0].subgridspec(2, 2, width_ratios=[3,1],wspace=0)
    gs01 = gs0[1].subgridspec(1,1)

    ax = []
    ax.append([fig.add_subplot(gs00[0, 0]),fig.add_subplot(gs00[0, 1])])
    ax.append([fig.add_subplot(gs00[1, 0]),fig.add_subplot(gs00[1, 1])])
    ax.append(fig.add_subplot(gs01[0, :]))


    ax[0][0].sharey(ax[0][1])
    ax[1][0].sharey(ax[1][1])
    
    ax[0][0].scatter(GP.training_indexes.flatten(),residuals,marker='.')
    ax[0][0].axhline(0,c='k',ls='--')
    ax[0][0].set_ylabel('Residuals')
    ax[0][0].set_xticklabels([])
        
    ax[1][0].scatter(GP.training_indexes.flatten(),scaled_residuals,c='C1',marker='.')
    ax[1][0].set_ylabel('Residuals / Error')
    ax[1][0].set_xlabel('Time')
    ax[1][0].axhline(0,c='k',ls='--')

    p = scipy.stats.norm.fit(scaled_residuals)

    ax[1][1].plot(scipy.stats.norm.pdf(np.linspace(-max_scale_res,max_scale_res,100),p[0],p[1]),np.linspace(-max_scale_res,max_scale_res,100))
    ax[1][1].hist(scaled_residuals,bins='auto',orientation='horizontal',density=True)
    ax[1][1].set_xticks([])
    ax[1][1].tick_params(axis='y',labelleft=False)

    ax[0][1].set_xticks([])
    ax[0][1].hist(residuals,bins='auto',orientation='horizontal',density=True)
    ax[0][1].tick_params(axis='both',labelleft=False,right='off', left='off',top='off', bottom='off')


    lag,acvf,l,b = ax[2].acorr(scaled_residuals,maxlags=maxlag,color='C2',linewidth=2,usevlines=True)
    axins = inset_axes(ax[2], width="60%", height="40%", loc='upper right')
    lag,acvf,l,b = axins.acorr(scaled_residuals,maxlags=10,color='C2',linewidth=2)
    axins.set_xlim(-.15,10)

    ax[2].set_xlim(-10,maxlag)
    ax[2].set_xlabel('Time lag')
    ax[2].set_ylabel('Autocorrelation\nResiduals / Error')
    axins.set_xlabel('Time lag')

    if title is not None:
        fig.suptitle(title)
    else:
        fig.suptitle('Residuals of the fit')
        
    fig.align_ylabels()
    fig.tight_layout()
    
    fig.savefig(f'{filename}_residuals.pdf')
    fig.show()
    