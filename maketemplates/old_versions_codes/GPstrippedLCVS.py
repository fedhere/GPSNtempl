import glob 
import os
import pylab as pl
import numpy as np
import scipy
import json
import sys
import pandas as pd
import scipy as sp
import numpy as np
from scipy import optimize
from scipy.interpolate import interp1d
from scipy import stats as spstats 
from scipy import integrate
import pickle as pkl

from scipy.interpolate import InterpolatedUnivariateSpline

from scipy.interpolate import UnivariateSpline,splrep, splev
import scipy.optimize as op
from scipy import interpolate

import george
from george import kernels
from george.kernels import ExpSquaredKernel


s = json.load( open(os.getenv ('PUI2015')+"/fbb_matplotlibrc.json") )
pl.rcParams.update(s)

PRETTY = True #False
if PRETTY:
     pl.rcParams["axes.facecolor"] =  "#FFFFFF"

cmd_folder = os.path.realpath(os.getenv("SESNCFAlib"))

if cmd_folder not in sys.path:
     sys.path.insert(0, cmd_folder)

import snclasses as snstuff
import templutils as templutils
import utils as snutils
import fitutils as fitutils
import myastrotools as myas
import matplotlib as mpl
import plotutils as plotutils

mpl.use('agg')


pl.rcParams['figure.figsize']=(10,10)


# # Loading CfA SN lightcurves

#setting parameters for lcv reader
#use literature data (if False only CfA data)
LIT=True
#use NIR data too
FNIR=True

def der(xy):
    xder,yder  = xy[1], xy[0]
    #print ("here ", yder[1] - yder[:-1])
    np.diff(yder) / np.diff(xder)
    return np.array([np.diff(yder) / np.diff(xder), xder[:-1] + np.diff(xder) * 0.5])

def nll(p, y, x0, x, gp, yerr):   
    gp.kernel[:] = p
    
    #try:
    #    gp.compute(x0, yerr)
    #except LinAlgError, ValueError:
    #    return 1e25
    smoothness = np.nansum(np.abs(der(der([gp.predict(y,x)[0], x]))), axis=1)[0]
    smoothness = smoothness if np.isfinite(smoothness) \
                 and ~np.isnan(smoothness) else 1e25
    ll = - gp.lnlikelihood(y, quiet=True) 
    ll +=  (smoothness)**1.5

    return ll if np.isfinite(ll) else -1e25

def grad_nll(p, y, x, gp):
    # Update the kernel parameters and compute the likelihood.
    smoothness = der(der([gp.predict(y,x)[0], x]))[0]
    # Update the kernel parameters and compute the likelihood.
    return -gp.grad_lnlikelihood(y, quiet=True) 


#uncomment for all lcvs to be read in
allsne = pd.read_csv(os.getenv("SESNCFAlib") +
                     "/SESNessentials.csv")['SNname'].values
if len(sys.argv)>1:
     thesn = sys.argv[1]
#set up SESNCfalib stuff
su = templutils.setupvars()
nbands = len(su.bands)

errorbarInflate = 1.#0.
#{"93J":10, 
#                   "05mf":1}

for sn in allsne:
    
    # read and set up SN and look for photometry files
    try:
         thissn = snstuff.mysn(sn, addlit=True)
    except AttributeError:
         continue

    if len(sys.argv)>1 and not thissn.snnameshort == thesn:
         continue

    
    if len(thissn.optfiles) + len(thissn.fnir) == 0:
        print ("bad sn")

    # read metadata for SN
    thissn.readinfofileall(verbose=False, earliest=False, loose=True)
    thissn.printsn()


    # check SN is ok and load data
    if thissn.Vmax is None or thissn.Vmax == 0 or np.isnan(thissn.Vmax):
        print ("bad sn")
    print (" starting loading ")
    print (os.environ['SESNPATH'] + "/finalphot/*" + \
                           thissn.snnameshort.upper() + ".*[cf]")
    print (os.environ['SESNPATH'] + "/finalphot/*" + \
                           thissn.snnameshort.lower() + ".*[cf]")
    
    print( glob.glob(os.environ['SESNPATH'] + "/finalphot/*" + \
                           thissn.snnameshort.upper() + ".*[cf]") + \
                 glob.glob(os.environ['SESNPATH'] + "/finalphot/*" + \
                           thissn.snnameshort.lower() + ".*[cf]") )   
    lc, flux, dflux, snname = thissn.loadsn2(verbose=False)
    thissn.setphot()
    thissn.getphot()
    thissn.setphase()
    thissn.sortlc()
    thissn.printsn()


    #check that it is k
    if np.array([n for n in thissn.filters.itervalues()]).sum() == 0:
            print ("bad sn")


    for b in su.bands:
        #if not b == 'U':
        #     continue
        if b in ['i','r','u']:
             bb = b+"p"
        else:
             bb = b
        if thissn.filters[b] == 0:
              continue

        xmin = thissn.photometry[b]['mjd'].min()
        x = thissn.photometry[b]['mjd'] - thissn.Vmax + 2400000.5
        y = thissn.photometry[b]['mag'] 
        y = y.min() - y
        yerr = thissn.photometry[b]['dmag'] * errorbarInflate
        
        if PRETTY:
             fig = pl.figure()#figsize=(20,20))
             ax00 = pl.subplot2grid((2, 2), (0, 0), colspan=1)
             ax01 = pl.subplot2grid((2, 2), (0, 1), colspan=1)             
             ax11 = pl.subplot2grid((2, 2), (1, 0), colspan=2)             
        else:
             fig = pl.figure()#figsize=(20,20))
             ax1 = fig.add_subplot(223)
             ax2 = fig.add_subplot(224)
             ax00 = fig.add_subplot(221)
             ax01 = fig.add_subplot(222)
             ax01.errorbar(x, y, yerr=yerr, fmt='k.')
             
             fig.suptitle("%s band %s"%(sn, b), fontsize=16)
             ax1.errorbar(x, y, yerr=yerr, fmt='k.')

        if not PRETTY:
             ax2.errorbar(np.log(x + 30), y, yerr=yerr, fmt='k.', lw=1)

        templatePkl = "UberTemplate_%s.pkl"%(b + 'p' if b in ['u', 'r', 'i']
                                                    else b)
        tmpl = pkl.load(open(templatePkl, "rb"))
        tmpl['mu'] = -tmpl['mu']
        print ("Template for the current band", templatePkl)

        t = np.linspace(x.min(), x.max(), 100)

        tmpl['musmooth'] = -tmpl['spl'](tmpl['phs'])
        meansmooth = lambda x : -tmpl['spl'](x) + tmpl['spl'](0)  

        if PRETTY:
             
             ax01.errorbar(np.log(x + 30), y + meansmooth(x),
                           yerr=yerr, lw=1, fmt='k.', alpha=0.5)
             ax11.errorbar(x, y + meansmooth(x), yerr=yerr,
                           fmt='k.', alpha=0.5, lw=1,
                           label=(thissn.snnameshort + " " +
                           bb.replace("p","'")))             
             ax11.legend(fontsize=15)
             
        
        ax00.errorbar(x, y, yerr=yerr, fmt='.', lw=1, label="data")
        ax00.plot([-25,95], [0,0], 'k-', alpha=0.5)
        ax00.plot(x, meansmooth(x), 'ko')
        ax00.plot(t, meansmooth(t), label="mean")
        ax00.plot(x, y - meansmooth(x), '.', color='DarkOrange',
                  label="residuals")
        ax00.legend(fontsize=15, ncol=2)
        ax00.set_ylim(-3,2)
        ax00.set_xlim(-30,120)
        ax00.set_ylabel("relative mag")
        ax00.set_xlabel("time (days since peak)")
        ax00.set_ylim(-3.3, 2.3)
        if not PRETTY:
             ax00.set_title("residuals")

        # Set up the Gaussian process.

        # ## Optimizing the hyper parameters

        kernel = kernels.Product(kernels.ConstantKernel(-1.15),
                                 kernels.ExpSquaredKernel(-2.3))

        gp = george.GP(kernel)
        gp.kernel[:] = (-1.10, -1.73)#(-1.15, -2.3) 

        p0 = gp.kernel.vector
        done = False
        try:
             gp.compute(np.log(x + 30), yerr)
             done = True
        except ValueError:
             continue
        '''        
        #else:
             k = -10
             while not done and k<3:
                  kernel = kernels.Product(kernels.ConstantKernel(np.sqrt(1e-1)),
                                           kernels.ExpSquaredKernel(10**(k)))

                  try:
                       gp = george.GP(kernel)
                       gp.compute(np.log(x + 30), yerr)                       
                  except ValueError:
                       k=k+0.3
                       continue
                  done = True
        if not done:
             continue
        '''
        mu, cov = gp.predict(y, np.log(t+30))
        std = np.sqrt(np.diag(cov))

        p0 = gp.kernel.vector

        if PRETTY:
             ax11.fill_between(t, mu-std  + meansmooth(t),
                               mu+std + meansmooth(t), color='grey', alpha=0.3)
             ax11.set_xlabel("time (days since peak)")
             ax11.plot(t, mu + meansmooth(t), 'IndianRed', lw=2)
             ax11.plot([0,0], ax11.get_ylim(), 'k-')
             moffset = snstuff.coffset[b]
             ax11.plot([0 + moffset, 0 + moffset], ax11.get_ylim(), 'k-', alpha=0.5)             
             ax01.plot(np.log(t+30), mu + meansmooth(t), 'IndianRed',
                       label="hyperparams: \n%.2f %.2f"%(p0[0], p0[1]), lw=2)
             ax01.fill_between(np.log(t+30), 
                        mu + meansmooth(t) - std, 
                        mu + meansmooth(t) + std , color='grey', alpha=0.3)
             ax01.legend(fontsize=15)
        
        else:
             ax01.set_title("pars: %.2f %.2f"%(p0[0], p0[1]))

             ax01.plot(t , mu, 'IndianRed', lw=2)
             ax01.fill_between(t, mu-std, mu+std, color='grey', alpha=0.3)
             ax01.set_title("pars: %.2f %.2f"%(p0[0], p0[1]))

        # # Subracting the mean and fitting GP to residuals only

        #spl = InterpolatedUnivariateSpline(templ.phs, ysmooth)
        #pl.savefig("GPfit%s_%s_medians.png"%(sn,b))

        if not PRETTY:#try:
            results = op.minimize(nll, p0, args=(y - meansmooth(x), 
                                                 x, np.log(t+30), gp, yerr))
            #    # Update the kernel and print the final log-likelihood.
            #if PLOT: pl.show()
            gp.kernel[:] = results.x
            #except :#RuntimeError:
            #pl.savefig("GPfit%s_%s.png"%(sn,b))
            #     pass #continue
            print ("hyper parameters: ", gp.kernel)
            print("loglikelihood", gp.lnlikelihood(y))

            gp.compute(np.log(x+30), yerr )
            mu, cov = gp.predict(y - meansmooth(x), np.log(t+30))
            std = np.sqrt(np.diag(cov))
            
            p1 = gp.kernel.vector        

        if PRETTY:
             ax11.set_ylabel("relative mag")             
             ax11.set_xlabel("time (days since peak)")
             ax01.set_xlabel("log time (peak-30 days)")                          
             print ("done with pretty")
        else:
             print ("here")
             ax1.set_xlabel("time (days since peak)")

             ax1.plot(t, mu + meansmooth(t), 'IndianRed', lw=2,
                 label="%.2f %.2f %s"%(p1[0], p1[1], results.success))
        
             ax1.fill_between(t,
                        mu + meansmooth(t) - std, 
                        mu + meansmooth(t) + std , color='grey', alpha=0.3)

             ax1.set_ylabel("relative mag")
             ax1.legend()
             ax1.set_ylim(ax01.get_ylim())
             ax1.set_xlim(ax01.get_xlim())        

             ax2.set_xlabel("log time (-30 days to peak)")

             ax2.plot(np.log(t+30), mu + meansmooth(t), 'r', lw=2)
             ax2.fill_between(np.log(t+30), 
                        mu + meansmooth(t) - std, 
                        mu + meansmooth(t) + std , color='grey', alpha=0.3)
        
        # # Subracting the mean and fitting GP to residuals only

        #spl = InterpolatedUnivariateSpline(templ.phs, ysmooth)



        if PRETTY:
             print ("GPfit%s_%s.pdf"%(sn,bb))
        
             pl.savefig("GPfit%s_%s.pdf"%(sn,bb))
        else:
             #ax1.xlabel("log time")             
             pl.savefig("GPfit%s_%s.png"%(sn,bb))             





