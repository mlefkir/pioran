
ICCF
====

.. py:module:: pioran.utils.ICCF

.. autoapi-nested-parse::

   Extracted from https://bitbucket.org/cgrier/python_ccf_code/src/master/
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

   ..
       !! processed by numpydoc !!


Overview
--------


.. list-table:: Function
   :header-rows: 0
   :widths: auto
   :class: summarytable

   * - :py:obj:`xcor <pioran.utils.ICCF.xcor>`\ (t1, y1, t2, y2, tlagmin, tlagmax, tunit, imode)
     - Calculate cross-correlation function for unevenly




Functions
---------
.. py:function:: xcor(t1, y1, t2, y2, tlagmin, tlagmax, tunit, imode=0)

   
   Calculate cross-correlation function for unevenly 
   sampling data.


   :Parameters:

       **t1** : array
           Time for light curve 1, assumed to be increasing.

       **y1** : array
           Flux for light curve 1.

       **t2** : array
           Time for light curve 2, assumed to be increasing.

       **y2** : array
           Flux for light curve 2.

       **tlagmin** : float
           Minimum time lag.

       **tlagmax** : float
           Maximum time lag.

       **tunit** : float
           Tau step.

       **imode** : int, optional
           Cross-correlation mode. Default is 0. Options are:
           0 - twice
           1 - interpolate light curve 1
           2 - interpolate light curve 2

   :Returns:

       **ccf** : array
           Correlation coefficient.

       **tlag** : array
           Time lag (t2 - t1). Positive values mean the second light curve lags the first light curve, as per convention. (edit by kate, march 2016)

       **npts** : int
           Number of data points used.













   ..
       !! processed by numpydoc !!




