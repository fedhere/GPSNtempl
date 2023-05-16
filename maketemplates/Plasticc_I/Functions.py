"""
V220317
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.table import Table
from collections import OrderedDict as odict
import time
from scipy.interpolate import interp1d
from numpy import math
from scipy.optimize import minimize
    
#Funtion read_snana_fits from https://sncosmo.readthedocs.io/en/v2.1.x/api/sncosmo.read_snana_fits.html
#Credit: Ming Lian repo: https://github.com/lmptc/PrestoColor2
def read_snana_fits(head_file, phot_file, snids=None, n=None):

    # Should we memmap? Only if we're going to read only a part of the file
    memmap = (snids is not None or n is not None)

    # Get metadata for all the SNe
    head_data = fits.getdata(head_file, 1, view=np.ndarray)
    phot_data = fits.getdata(phot_file, 1, view=np.ndarray, memmap=memmap)

    # Strip trailing whitespace characters from SNID.
    if 'SNID' in head_data.dtype.names:
        try:
            head_data['SNID'][:] = np.char.strip(head_data['SNID'])
        except TypeError:
            pass

    # Check which indicies to return.
    if snids is None and n is None:
        idx = range(len(head_data))
    elif n is None:
        if 'SNID' not in head_data.dtype.names:
            raise RuntimeError('Specific snids requested, but head file does'
                               ' not contain SNID column')
        idx = []
        for snid in snids:
            i = np.flatnonzero(head_data['SNID'] == snid)
            if len(i) != 1:
                raise RuntimeError('Unique snid requested, but there are '
                                   '{0:d} matching entries'.format(len(i)))
            idx.append(i[0])
    elif snids is None:
        idx = range(n)
    else:
        raise ValueError("cannot specify both 'snids' and 'n' arguments")

    # Loop over SNe in HEAD file
    sne = []
    for i in idx:
        meta = odict(zip(head_data.dtype.names, head_data[i]))

        j0 = head_data['PTROBS_MIN'][i] - 1
        j1 = head_data['PTROBS_MAX'][i]
        data = phot_data[j0:j1]
        if 'FLT' in data.dtype.names:
            data['FLT'][:] = np.char.strip(data['FLT'])
        sne.append(Table(data, meta=meta, copy=False))

    return sne

def exp_fit(t, t0, shift):
    # exp_rise = np.ones(len(t))
    exp_rise = np.e**(np.asarray(shift * (t - t0))) + 1
    return exp_rise

def sn_fit(t, gmax, gmean, gwidth, t0, shift, dec_slope, dec_inter):

    gaussian = gmax * np.exp(-1 * ((t - gmean)**2) / (gwidth**2))
    # point1 =  gmax * np.exp(-1 * ((cutoff)**2) / 2 * (gwidth**2))
    # dec_inter = point1 - dec_slope * cutoff
    linear = np.zeros(len(t))
    if t[-1] > 20:
        linear = dec_slope * t + dec_inter
    # print(linear, gaussian, dec_inter)
    return exp_fit(t, t0, shift) * (gaussian + linear)

def nll_VC(theta, t, f, f_err):
    gmax, gmean, gwidth, t0, shift, dec_slope, dec_inter = theta
    model = sn_fit(t, gmax, gmean, gwidth, t0, shift, dec_slope, dec_inter)
    inv_sigma2 = 1.0/(f_err**2)
    return (np.sum((f-model)**2*inv_sigma2))

def der(xy):
    xder,yder  = xy[1], xy[0]
    #print ("here ", yder[1] - yder[:-1])
    np.diff(yder) / np.diff(xder)
    return np.array([np.diff(yder) / np.diff(xder), xder[:-1] + np.diff(xder) * 0.5])

def nll_gp(p, y, x, gp, s):
    # gp.kernel.parameter_vector = p


    gp.set_parameter_vector(p)
    # print(gp.get_parameter_vector)
    # print('lengths: ',len(np.log(xx + 30)), len(yerr))
    # gp.compute(np.log(xx + 30), yerr)

    # Calculate smoothness of the fit
    # pred = gp.predict(y, x)[0]
    # print(pred)
    # print(x)
    try:
     smoothness = (np.nansum(np.abs(der(der([gp.predict(y,x)[0], x]))), axis=1)[0])
     smoothness = smoothness if np.isfinite(smoothness) \
                 and ~np.isnan(smoothness) else 1e25    
    except np.linalg.LinAlgError:
       smoothness =  1e25
    
    # print(gp.log_likelihood(y, squiet=True), smoothness)

    ll = gp.log_likelihood(y, quiet=True)  #- (smoothness) #np.sum((y - pred[inds]**2)) #
    ll -= (smoothness)** s

    # print (p, -ll if np.isfinite(ll) else 1e25)
    return -ll if np.isfinite(ll) else 1e25

def Chebyhev_fitter (df, degree):
    n = degree

    if n <11:
        print('Degree must be more than 10.')
        sys.exit()
    xmin = min(df['t'])
    xmax = max(df['t'])
    bma = 0.5 * (xmax - xmin)
    bpa = 0.5 * (xmax + xmin)
    interpoll = interp1d(df['t'], df['A'], kind='cubic')
    f = [interpoll(math.cos(math.pi * (k + 0.5) / n) * bma + bpa) for k in range(n)]
    fac = 2.0 / n
    cheby_coefficients = [fac * sum([f[k] * math.cos(math.pi * j * (k + 0.5) / n) for k in range(n)]) for j in range(n)]


    Cheby_func = []

    for t_i in np.sort(df['t'].values):

        y = (2.0 * t_i - xmin - xmax) * (1.0 / (xmax - xmin))
        y2 = 2.0 * y
        (d, dd) = (cheby_coefficients[-1], 0)             # Special case first step for efficiency

        for cj in cheby_coefficients[-2:0:-1]:            # Clenshaw's recurrence
            (d, dd) = (y2 * d - dd + cj, d)
        Cheby_func.append(y * d - dd + 0.5 * cheby_coefficients[0])

    Cheby_func = np.asarray(Cheby_func)

    return Cheby_func

def lc_fit(lcv, x_peak = 'None'):
    # lcv should be a numpy array with three columns: mjd, flux, flux error

    t = lcv[0]
    f = lcv[1]
    ferr = lcv[2]

    t = t[f > 0]
    ferr = ferr[f > 0]
    f = f[f > 0]

    if len(f) == 0:
        # print('No data was found!')
        sys.exit()

    m = 27.5 - 2.5 * np.log10(f)  # Zero point magnitude for ELASTICC light curves = 27.5
    median = np.nanmedian(m)
    m = m - median
    merr = 2.5 / np.log(10) * ferr / f

    if x_peak == 'None':
        x_peak = t[np.where(m == min(m))[0]]
    else:
        x_peak = x_peak

    y_peak = m[np.argmin(m)]

    low_lim = -50
    up_lim = 100
    x = t - x_peak
    ind = (x < up_lim) & (x > low_lim)

    y = m[ind]
    yerr = merr[ind]
    f = f[ind]
    ferr = ferr[ind]
    x = x[ind]
    y = y - y[np.argmin(y)]
    t_new = np.linspace(x.min(), x.max(), 1000)

    # Set up initial parameters

    p0 = [0] * 7
    p0[0] = 4
    peakdate = 0#t[np.where(m == min(m))[0]]

    # if len(peakdate) > 1:
    #     peakdate = peakdate[0]

    p0[1] = peakdate + 5
    p0[2] = 10  # sigma

    lintail = np.where(x > peakdate + 50)[0]
    if len(lintail) < 1:
        # print("no tail data")
        if np.diff(x[-2:])[0]>1:
            linfit = np.polyfit(x[-2:], f[-2:], 1)
        else:
            linfit = np.polyfit(x[-3:], f[-3:], 1)
        p0[6] = linfit[1]
        p0[5] = linfit[0]
    else:
        linfit = np.polyfit(x[lintail], f[lintail], 1)
        p0[6] = linfit[1]
        p0[5] = linfit[0]

    p0[4] = 0.1
    p0[3] = peakdate - 20

    res1 = minimize(nll_VC, p0,
                    args=(np.asarray(x), np.asarray(f), np.asarray(ferr)), method='Powell')

    return t_new, sn_fit(t_new, *res1['x']), res1.x