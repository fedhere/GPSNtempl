

import pylab as pl
import numpy as np
import george
from george import kernels
from george.kernels import ExpSquaredKernel

import glob 
import inspect
import optparse
import time
import copy
import os
import pylab as pl
import numpy as np
import scipy
import json
import sys
import pickle as pkl

import scipy as sp
import numpy as np
from scipy import optimize
from scipy.interpolate import interp1d
from scipy import stats as spstats 
from scipy import integrate

from scipy.interpolate import InterpolatedUnivariateSpline

from scipy.interpolate import UnivariateSpline,splrep, splev
import scipy.optimize as op
from scipy import interpolate

import multiprocessing as mpc

import json
import os
import pandas as pd

s = json.load( open(os.getenv ('PUI2015')+"/fbb_matplotlibrc.json") )
pl.rcParams.update(s)

cmd_folder = os.path.realpath(os.getenv("SESNCFAlib"))

if cmd_folder not in sys.path:
     sys.path.insert(0, cmd_folder)

import snclasses as snstuff
import templutils as templutils
import utils as snutils
import fitutils as fitutils
import myastrotools as myas
import matplotlib as mpl

mpl.use('agg')

import pylab as pl
from pylab import rc
import plotutils as plotutils
from scipy.interpolate import interp1d
import templutils as tpl

pl.rcParams['figure.figsize']=(10,10)


# # Loading CfA SN lightcurves

#setting parameters for lcv reader
#use literature data (if False only CfA data)
LIT=True
#use NIR data too
FNIR=True


def der(xy):
    x,y  = xy[0],xy[1]
    return [(y[1]-y[:-1] ) / np.diff(y)[0],  x ] 

def nll(p, y, x, gp):
    # Update the kernel parameters and compute the likelihood.
    gp.kernel[:] = p
    smoothness = np.nansum(np.abs(der(der([gp.predict(y,x)[0], x]))),axis=1)[0]
    #print ("here", smoothness)
    smoothness = smoothness if np.isfinite(smoothness) and ~np.isnan(smoothness) else 1e25
    ll = gp.lnlikelihood(y, quiet=True)  + smoothness
    return -ll if np.isfinite(ll) else 1e25

def grad_nll(p, y, x, gp):
    # Update the kernel parameters and compute the likelihood.
    gp.kernel[:] = p
    smoothness = np.nansum(np.abs(der(der([gp.predict(y,x)[0], x]))),axis=1)[0]
    #print ("here", smoothness)
    smoothness = smoothness if np.isfinite(smoothness) and ~np.isnan(smoothness) else 1e25
    return -gp.grad_lnlikelihood(y, quiet=True) - der(smoothness)

#uncomment for all lcvs to be read in
allsne = pd.read_csv(os.getenv("SESNCFAlib") + "/SESNessentials.csv")['SNname'].values

#set up SESNCfalib stuff
su = templutils.setupvars()
nbands = len(su.bands)

#errorbarInflate = {"93J":30, 
#                   "05mf":1}

for sn in allsne[10:]:

    # read and set up SN and look for photometry files
    try:
         thissn = snstuff.mysn(sn, addlit=True)
    except AttributeError:
         continue
    if len(thissn.optfiles) + len(thissn.fnir) == 0:
        print ("bad sn")

    # read metadata for SN
    thissn.readinfofileall(verbose=True, earliest=False, loose=True)
    thissn.printsn()


    # check SN is ok and load data
    if thissn.Vmax is None or thissn.Vmax == 0 or np.isnan(thissn.Vmax):
        print ("bad sn")
    print (" starting loading ")    
    lc, flux, dflux, snname = thissn.loadsn2(verbose=True)
    thissn.setphot()
    thissn.getphot()
    thissn.setphase()
    thissn.sortlc()
    thissn.printsn()


    #check that it is k
    if np.array([n for n in thissn.filters.itervalues()]).sum() == 0:
            print ("bad sn")


    for b in su.bands:
        if thissn.filters[b] == 0:
              continue

        print(thissn.Vmax, 2400000.5)
        xmin = thissn.photometry[b]['mjd'].min()
        x = thissn.photometry[b]['mjd'] - thissn.Vmax + 2400000.5
        y = thissn.photometry[b]['mag'] 
        y = y.min() - y
        print(x,y)
        yerr = thissn.photometry[b]['dmag']
        
        fig = pl.figure(figsize=(20,10))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        fig.suptitle("%s band %s"%(sn, b), fontsize=16)
        ax1.errorbar(x, y, yerr=yerr, fmt='k.')
        ax2.errorbar(np.log(x + 30), y, yerr=yerr, fmt='k.')

        templatePkl = "UberTemplate_%s.pkl"%(b + 'p' if b in ['u', 'r', 'i']
                                                    else b)
        tmpl = pkl.load(open(templatePkl, "rb"))
        tmpl['mu'] = -tmpl['mu']
        print ("Template for the current band", templatePkl)


        tmpl['musmooth'] = -tmpl['spl'](tmpl['phs'])
        meansmooth = lambda x : -tmpl['spl'](x) + tmpl['spl'](0)  


        # Set up the Gaussian process.

        # ## Optimizing the hyper parameters

        t = np.linspace(x.min(), x.max(), 100)
        kernel = kernels.Product(kernels.ConstantKernel(np.sqrt(1e-1)),
                                 kernels.ExpSquaredKernel(0.1))

        gp = george.GP(kernel)

        p0 = gp.kernel.vector
        done = False
        try:
             gp.compute(np.log(x + 30), yerr)
        except ValueError:
             k = -3
             while not done and k<3:
                  kernel = kernels.Product(kernels.ConstantKernel(np.sqrt(1e-1)),
                                           kernels.ExpSquaredKernel(10**(k)))

                  try:
                       gp = george.GP(kernel)
                       print(k)
                  except ValueError:
                       k=k+0.3
                       continue
                  done = True
        if not done: continue


        try:
             results = op.minimize(nll, p0, jac=grad_nll, args=(y - meansmooth(x),
                                                           np.log(t+30), gp))
            #    # Update the kernel and print the final log-likelihood.
        except RuntimeError:
             pl.savefig("GPfit%s_%s.png"%(sn,b))
             continue
        gp.kernel[:] = results.x
        print ("hyper parameters: ",gp.kernel)
        print("loglikelihood", gp.lnlikelihood(y))

        gp.compute(np.log(x+30), yerr )
        mu, cov = gp.predict(y, np.log(t+30))
        std = np.sqrt(np.diag(cov))


        ax1.set_title("SN %s band %s"%(sn, b))
        pl.ylabel("normalized mag")
        ax1.set_xlabel("time (days since peak)")
        ax2.set_xlabel("log time (days since peak)")

        ax1.set_ylabel("normalized mag")
        ax2.set_ylabel("normalized mag")
        ax1.plot(t , mu, 'r', lw=2)
        ax1.fill_between(t, mu-std, mu+std, color='grey', alpha=0.3)
        ax2.plot(np.log(t+30), mu + meansmooth(t), 'r', lw=2)
        ax2.fill_between(np.log(t+30), 
                        mu + meansmooth(t) - std, 
                        mu + meansmooth(t) + std , color='grey', alpha=0.3)

        # # Subracting the mean and fitting GP to residuals only

        #spl = InterpolatedUnivariateSpline(templ.phs, ysmooth)


        pl.ylabel("normalized mag")
        xl = pl.xlabel("log time (starting 30 days before peak)")
        pl.savefig("GPfit%s_%s.png"%(sn,b))





