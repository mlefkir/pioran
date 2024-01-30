"""Extracted from https://bitbucket.org/cgrier/python_ccf_code/src/master/
Version 1, by Mouyuan (Eric) Sun
email: ericsun88@live.com
Version 2, by Kate Grier
email: catherine.grier@gmail.com

Copyright (c) 2018 Mouyuan Sun and Catherine Grier; catherine.grier@gmail.com  

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.



This code is meant to emulate a fortran program written by B. Peterson,
which cross correlates two light curves that are unevenly sampled using linear
interpolation and measure sthe peak and centroid of the cross-correlation function.
In addition, it is possible to run Monteo Carlo iterationsusing flux randomization
and random subset selection (RSS) to produce cross-correlation centroid distributions to
estimate the uncertainties in the cross correlation results.

The idea is described in detail in this work:
Peterson et al.(1998): http://arxiv.org/abs/astro-ph/9802103

The modules included in this code are:

2. xcor: Calculates the cross correlation function (called by peakcent and xcor_mc). Again does not
need to be used -- is called by the below two modules. 

Version history:
Version 1.0 (May 27, 2015)
Version 1.1 (Nov 15, 2015):
Bug fix: make sure the peak is significant when calculate the centroid (i.e., add ' and status_peak==1 ' to line 212)
Version 2.0: (Jan 5, 2018)
This version of the software hasbeen edited fairly extensively by Kate Grier and Jennifer Li
from its original version, to fix bugs and add additional functionality.

"""

import numpy as np

def xcor(t1, y1, t2, y2, tlagmin, tlagmax, tunit, imode=0):
    """
    Calculate cross-correlation function for unevenly 
    sampling data.
    
    
    Parameters
    ----------
    t1 : array
        Time for light curve 1, assumed to be increasing.
    y1 : array
        Flux for light curve 1.
    t2 : array
        Time for light curve 2, assumed to be increasing.
    y2 : array
        Flux for light curve 2.
    tlagmin : float
        Minimum time lag.
    tlagmax : float
        Maximum time lag.
    tunit : float
        Tau step.
    imode : int, optional
        Cross-correlation mode. Default is 0. Options are:
        0 - twice
        1 - interpolate light curve 1
        2 - interpolate light curve 2

    Returns
    -------
    ccf : array
        Correlation coefficient.
    tlag : array
        Time lag (t2 - t1). Positive values mean the second light curve lags the first light curve, as per convention. (edit by kate, march 2016)
    npts : int
        Number of data points used.
    """
    safe = tunit*0.1
    taulist = []
    npts = []
    ccf12 = []  # interpolate 2
    ccf21 = []  # interpolate 1
    
    # frist interpolate 2
    tau = tlagmin + 0.0
    while tau < tlagmax+safe:
        t2new = t1 - tau
        selin = np.where((t2new>=np.min(t2))&(t2new<=np.max(t2)), True, False)
        knot = np.sum(selin)  # number of datapoints used
        if knot>0:
            y2new = np.interp(t2new[selin], t2, y2)
            
            y1sum = np.sum(y1[selin])
            y1sqsum = np.sum(y1[selin]*y1[selin])
            y2sum = np.sum(y2new)
            y2sqsum = np.sum(y2new*y2new)
            y1y2sum = np.sum(y1[selin]*y2new)
            
            fn = float(knot)
            rd1_sq = fn*y2sqsum - y2sum*y2sum
            rd2_sq = fn*y1sqsum - y1sum*y1sum
            if rd1_sq>0.0:
                rd1 = np.sqrt(rd1_sq)
            else:
                rd1 = 0.0
            if rd2_sq>0.0:
                rd2 = np.sqrt(rd2_sq)
            else:
                rd2 = 0.0
            
            if rd1*rd2==0.0:
                r = 0.0
            else:
                r = (fn*y1y2sum - y2sum*y1sum)/(rd1*rd2)
            ccf12.append(r)
            taulist.append(tau)
            npts.append(knot)
        tau += tunit
    # now interpolate 1
    tau = tlagmin + 0.0
    while tau < tlagmax+safe:
        t1new = t2 + tau
        selin = np.where((t1new>=np.min(t1))&(t1new<=np.max(t1)), True, False)
        knot = np.sum(selin)  # number of datapoints used
        if knot>0:
            y1new = np.interp(t1new[selin], t1, y1)
            
            y2sum = np.sum(y2[selin])
            y2sqsum = np.sum(y2[selin]*y2[selin])
            y1sum = np.sum(y1new)
            y1sqsum = np.sum(y1new*y1new)
            y1y2sum = np.sum(y1new*y2[selin])
            
            fn = float(knot)
            rd1_sq = fn*y2sqsum - y2sum*y2sum
            rd2_sq = fn*y1sqsum - y1sum*y1sum
            if rd1_sq>0.0:
                rd1 = np.sqrt(rd1_sq)
            else:
                rd1 = 0.0
            if rd2_sq>0.0:
                rd2 = np.sqrt(rd2_sq)
            else:
                rd2 = 0.0
            
            if rd1*rd2==0.0:
                r = 0.0
            else:
                r = (fn*y1y2sum - y2sum*y1sum)/(rd1*rd2)
            ccf21.append(r)
        tau += tunit
    
    # return results according to imode
    taulist = np.asarray(taulist)
    npts = np.asarray(npts)
    ccf12 = np.asarray(ccf12)
    ccf21 = np.asarray(ccf21)
    if imode==0:
        ccf = (ccf12 + ccf21)*0.5
    elif imode==1:
        ccf = ccf21 + 0.0
    else:
        ccf = ccf12 + 0.0
    
    return ccf, -taulist, npts

