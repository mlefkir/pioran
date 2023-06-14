"""Collection of functions for plotting the results of the Gaussian Process Regression.

"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import lombscargle
from scipy.stats import norm
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from .utils.ICCF import xcor


def plot_prediction(x,y,yerr,x_pred,y_pred,cov_pred,filename,figsize=(16,6),confidence_bands=True,title=None,xlabel=r'Time',ylabel=None,xlim=None,ylim=None,**kwargs):
    """Plot the predicted time series and the confidence bands.

    Parameters
    ----------
    x : :obj:`numpy.ndarray`
        Time of the observations.
    y : :obj:`numpy.ndarray`
        Values of the observations.
    yerr : :obj:`numpy.ndarray`
        Error of the observations.
    x_pred : :obj:`numpy.ndarray`
        Time of the prediction.
    y_pred : :obj:`numpy.ndarray`
        Values of the prediction.
    cov_pred : :obj:`numpy.ndarray`
        Covariance matrix of the prediction.    
    filename : :obj:`str`
        Name of the file to save the figure.
    figsize : :obj:`tuple`, optional
        Size of the figure.
    confidence_bands : :obj:`bool`, optional
        Plot the confidence bands, by default True
    title : :obj:`str`, optional
        Title of the plot, by default None
    xlabel : :obj:`str`, optional
        Label for the x-axis, by default None
    ylabel : :obj:`str`, optional
        Label for the y-axis, by default None
    xlim : :obj:`tuple` of :obj:`float`, optional
        Limits of the x-axis, by default None
    ylim : :obj:`tuple` of :obj:`float`, optional
        Limits of the y-axis, by default None
        
    Returns
    -------
    fig : :obj:`matplotlib.figure.Figure`
        Figure object.
    ax : :obj:`matplotlib.axes.Axes`
        Axes object.
    """  
    show = kwargs.get('show',False)

    fig,ax = plt.subplots(1,1,figsize=figsize)
     
    if confidence_bands:
        std = np.sqrt(np.diag(cov_pred))
        hi = (y_pred-std)
        lo = (y_pred+std)
        hi2 = (y_pred-2*std)
        lo2 = (y_pred+2*std)
        ax.fill_between(x_pred,hi2,lo2,alpha=0.25,label=r"$2\sigma$")
        ax.fill_between(x_pred,hi,lo,alpha=0.5,label=r"$1\sigma$")
    
    ax.errorbar(x ,y, yerr=yerr, fmt='.', label='Observation')
    ax.plot(x_pred,y_pred, label='Prediction',color='k')

    ax.set_title(title) if title is not None else None
    ax.set_xlabel(xlabel) if title is not None else None
    ax.set_ylabel(ylabel) if title is not None else None
    ax.set_xlim(xlim) if xlim is not None else None
    ax.set_ylim(ylim) if ylim is not None else None
    

    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    ax.legend()
    
    fig.tight_layout()
    fig.savefig(f"{filename}_prediction.pdf",bbox_inches='tight')
    if show:
        fig.show()
    return fig,ax
    
def plot_residuals(x,y,yerr,y_pred,filename,confidence_intervals=[95,99],figsize=(10,10),maxlag=None,title=None,**kwargs):
    """Plot the residuals of a predicted time series.


    Parameters
    ----------
    x : :obj:`numpy.ndarray`
        Time of the observations.
    y : :obj:`numpy.ndarray`
        Values of the observations.
    yerr : :obj:`numpy.ndarray`
        Error of the observations.
    y_pred : :obj:`numpy.ndarray`
        Values of the prediction at the time of the observations.
    filename : :obj:`str`
        Name of the file to save the figure.
    confidence_intervals : :obj:`list` of :obj:`float`, optional
        Confidence intervals to plot, by default [95,99]
    figsize : :obj:`tuple`, optional
        Size of the figure.
    maxlag : :obj:`int`, optional
        Maximum lag to plot, by default None
    title : :obj:`str`, optional
        Title of the plot, by default None
        
    Returns
    -------
    fig : :obj:`matplotlib.figure.Figure`
        Figure object.
    ax : :obj:`matplotlib.axes.Axes`
        Axes object.
    """

    show = kwargs.get('show',False)
    
    if maxlag is None:
        maxlag = len(x)-2
    else:
        maxlag = np.rint(maxlag)

    n = len(x)
    residuals = y - y_pred
    scaled_residuals = (residuals/yerr).flatten()
    max_scale_res = np.rint(np.max(scaled_residuals))
    sigs = [norm.ppf((50+ci/2)/100) for ci in confidence_intervals]

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
    
    ax[0][0].scatter(x,residuals,marker='.')
    ax[0][0].axhline(0,c='k',ls='--')
    ax[0][0].set_ylabel('Residuals')
    ax[0][0].set_xticklabels([])
        
    ax[1][0].scatter(x,scaled_residuals,c='C1',marker='.')
    ax[1][0].set_ylabel('Residuals / Error')
    ax[1][0].set_xlabel('Time')
    ax[1][0].axhline(0,c='k',ls='--')

    p = norm.fit(scaled_residuals)

    ax[1][1].plot(norm.pdf(np.linspace(-max_scale_res,max_scale_res,100),p[0],p[1]),np.linspace(-max_scale_res,max_scale_res,100))
    ax[1][1].hist(scaled_residuals,bins='auto',orientation='horizontal',density=True,alpha=0.75)
    ax[1][1].set_xticks([])
    ax[1][1].tick_params(axis='y',labelleft=False)

    ax[0][1].set_xticks([])
    ax[0][1].hist(residuals,bins='auto',orientation='horizontal',density=True,alpha=0.75)
    ax[0][1].tick_params(axis='both',labelleft=False,right='off', left='off',top='off', bottom='off')


    lag,acvf,l,b = ax[2].acorr(scaled_residuals,maxlags=maxlag,color='C2',linewidth=2,usevlines=True)
    
    axins = inset_axes(ax[2], width="60%", height="40%", loc='upper right')
    lag,acvf,l,b = axins.acorr(scaled_residuals,maxlags=10,color='C2',linewidth=2)
    for i,sig in enumerate(reversed(sigs)):
        ax[2].fill_between(np.linspace(0,maxlag,100),sig/np.sqrt(n),-sig/np.sqrt(n),color='C2',alpha=.15*(i+1),label=f'{confidence_intervals[i]}%')
        axins.fill_between(np.linspace(0,10,100),sig/np.sqrt(n),-sig/np.sqrt(n),color='C2',alpha=.15*(i+1))
    axins.set_xlim(0,10)


    ax[2].set_xlim(0,maxlag)
    ax[2].set_xlabel('Time lag')
    ax[2].margins(x=0,y=0)
    axins.margins(x=0,y=0)
    ax[2].set_ylabel('Autocorrelation\nResiduals / Error')
    ax[2].legend(loc='upper left')
    axins.set_xlabel('Time lag')

    if title is not None:
        fig.suptitle(title)
    else:
        fig.suptitle('Residuals of the fit')
        
    fig.align_ylabels()
    fig.tight_layout()
    
    fig.savefig(f'{filename}_residuals.pdf',bbox_inches='tight')
    if show:
        fig.show()
    return fig,ax

def plot_posterior_predictive_ACF(tau,acf,x,y,filename,with_mean=False,confidence_bands=[68,95],xlabel='Time lag (d)',save_data=False,**kwargs):
    """Plot the posterior predictive Autocorrelation function of the process.

    This function will also compute the interpolated cross-correlation function using the 
    code from https://bitbucket.org/cgrier/python_ccf_code/src/master/   

    Parameters
    ----------
    tau : :obj:`numpy.ndarray`
        Time lags.
    acf : :obj:`numpy.ndarray`
        Array of ACFs posterior samples.
    x : :obj:`numpy.ndarray`
        Time indexes.
    y : :obj:`numpy.ndarray`
        Time series values.
    filename : :obj:`str`
        Filename to save the figure.
    with_mean : bool, optional
        Plot the mean of the samples, by default False
    confidence_bands : list, optional
        Confidence intervals to plot, by default [95,99]
    xlabel : str, optional
        , by default r'Time lag (d)'
    save_data : bool, optional
        Save the data to a text file, by default False

    Returns
    -------
    fig : :obj:`matplotlib.figure.Figure`
        Figure object.
    ax : :obj:`matplotlib.axes.Axes`
        Axes object.
    """


    percentiles = jnp.sort(jnp.hstack(((50-np.array(confidence_bands)/2,50+np.array(confidence_bands)/2))))

    fig,ax = plt.subplots(figsize=(9,5))
    
    acf_median = jnp.median(acf,axis=0)
    ax.plot(tau,acf_median,c='C0',label='Median')
    
    acf_quantiles = jnp.percentile(acf,q=percentiles,axis=0)
    for i,ci in enumerate(confidence_bands):
        ax.fill_between(tau,acf_quantiles[i],acf_quantiles[-(i+1)],color='C0',alpha=.15*(i+1),label=f'{ci}%')
    
    if with_mean:
        acf_mean = jnp.mean(acf,axis=0)
        ax.plot(tau,acf_mean,label='Mean',ls='--')
    
    # Compute the ICCF of the data
    ccf, taulist, npts = xcor(x,y,x,y,tlagmin=0,tlagmax=x[-1]/2,tunit=np.mean(np.diff(x))/2,imode=0)
    
    if save_data:
        np.savetxt(f'{filename}_posterior_predictive_ACF.txt',jnp.vstack([tau,acf_median,acf_quantiles]),header='tau,acf_median,acf_quantiles')
        np.savetxt(f'{filename}_ICCF.txt',jnp.vstack([taulist,ccf]),header='taulist,ccf')    
    ax.plot(-taulist,ccf,label='ICCF',c='C2')   
    ax.legend()
    ax.margins(x=0,y=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r'Autocorrelation function')
    ax.set_title('Posterior predictive autocorrelation function')
    fig.tight_layout()
    fig.savefig(f'{filename}_posterior_predictive_ACF.pdf',bbox_inches='tight')
    return fig,ax


def plot_posterior_predictive_PSD(f,posterior_PSD,x,y,yerr,filename,save_data=False,with_mean=False,confidence_bands=[68,95],ylim=None,xlabel=r'Frequency $\mathrm{d}^{-1}$'):
    """Plot the posterior predictive Power Spectral Density of the process.

    This function will also compute the Lomb-Scargle periodogram on the data.

    Parameters
    ----------
    f : :obj:`numpy.ndarray`
        Frequencies.
    posterior_PSD : :obj:`numpy.ndarray`
        Array of PSDs posterior samples.
    x : :obj:`numpy.ndarray`
        Time indexes.
    y : :obj:`numpy.ndarray`
        Time series values.
    yerr : :obj:`numpy.ndarray`
        Time series errors.  
    filename : :obj:`str`
        Filename to save the figure.
    with_mean : bool, optional
        Plot the mean of the samples, by default False
    confidence_bands : list, optional
        Confidence intervals to plot, by default [95,99]
    xlabel : str, optional
        , by default r'Time lag (d)'
    save_data : bool, optional
        Save the data to a text file, by default False

    Returns
    -------
    fig : :obj:`matplotlib.figure.Figure`
        Figure object.
    ax : :obj:`matplotlib.axes.Axes`
        Axes object.
    """
    percentiles = jnp.sort(jnp.hstack(((50-np.array(confidence_bands)/2,50+np.array(confidence_bands)/2))))
    fig,ax = plt.subplots(figsize=(10,5))
    
    psd_median = jnp.median(posterior_PSD,axis=0)
    ax.loglog(f,psd_median,c='C0',label='Median')
    
    PSD_quantiles = jnp.percentile(posterior_PSD,q=percentiles,axis=0)
    for i,ci in enumerate(confidence_bands):
        ax.fill_between(f,PSD_quantiles[i],PSD_quantiles[-(i+1)],color='C0',alpha=.15*(i+1),label=f'{ci}%')
    
    
    if with_mean:
        psd_mean = jnp.mean(posterior_PSD,axis=0)
        ax.loglog(f,psd_mean,label='Mean',ls='--')
        
    # compute the Lomb-Scargle periodogram
    LS_periodogram = lombscargle(x,y,2*np.pi*f,precenter=True)    
    ax.loglog(f,LS_periodogram,color='C2',label='Lomb-Scargle')
    
    noise_level = np.median(np.diff(x))*np.mean(yerr**2)*2
    ax.axhline(noise_level,color='k',ls='--',label='Noise level')
    
    if save_data:
        np.savetxt(f'{filename}_posterior_predictive_PSD.txt',jnp.vstack((f,psd_median,PSD_quantiles,LS_periodogram)).T,
               header=f'f,psd_median,psd_quantiles({percentiles}),LS_periodogram')
    
    if ylim is None:
        # ax.set_ylim(bottom=np.min(LS_periodogram)/1e3)
        ax.set_ylim(bottom=noise_level/5)
        
    ax.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
    ax.margins(x=0,y=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r'Power spectral density')
    ax.set_title('Posterior predictive power spectral density')
    fig.tight_layout()
    fig.savefig(f'{filename}_posterior_predictive_PSD.pdf',bbox_inches='tight')
    return fig,ax