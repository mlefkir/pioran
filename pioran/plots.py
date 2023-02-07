"""Collection of functions for plotting the results of the Gaussian Process Regression.

"""

import numpy as np
import matplotlib.pyplot as plt    
from matplotlib.ticker import AutoMinorLocator
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

def plot_prediction(GP: GaussianProcess,filename,figsize,confidence_bands=True,title=None,xlabel=None,ylabel=None,xlim=None,ylim=None):
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
    fig.savefig(f"{filename}.pdf")