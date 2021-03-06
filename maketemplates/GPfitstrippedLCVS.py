import glob 
import os
import pylab as pl
import numpy as np
import scipy
import json
import sys
import pickle as pkl
import pandas as pd
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

import george
from george import kernels
from george.kernels import ExpSquaredKernel


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
import plotutils as plotutils

mpl.use('agg')


pl.rcParams['figure.figsize']=(10,10)


# # Loading CfA SN lightcurves

#setting parameters for lcv reader
#use literature data (if False only CfA data)
LIT = True
#use NIR data too
FNIR = True
FITGP = False #True

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

if __name__ == '__main__':
     #uncomment for all lcvs to be read in
     if len(sys.argv) > 1:
          allsne = [sys.argv[1]]
     else:
          allsne = pd.read_csv(os.getenv("SESNCFAlib") +
                          "/SESNessentials.csv")['SNname'].values
     print (allsne)

          
     #set up SESNCfalib stuff
     su = templutils.setupvars()
     nbands = len(su.bands)

     #errorbarInflate = {"93J":30, 
     #                   "05mf":1}

     for sn in allsne:
          
          # read and set up SN and look for photometry files
          try:
               thissn = snstuff.mysn(sn, addlit=True)
          except AttributeError:
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
               if thissn.filters[b] == 0:
                    continue

               xmin = thissn.photometry[b]['mjd'].min()
               x = thissn.photometry[b]['mjd'] - thissn.Vmax + 2400000.5
               y = thissn.photometry[b]['mag'] 
               y = y.min() - y
               yerr = thissn.photometry[b]['dmag']
        
               fig = pl.figure()#figsize=(20,20))
               ax1 = fig.add_subplot(223)
               ax2 = fig.add_subplot(224)
               ax00 = fig.add_subplot(221)
               ax01 = fig.add_subplot(222)        
               fig.suptitle("%s band %s"%(sn, b), fontsize=16)
               ax01.errorbar(x, y, yerr=yerr, fmt='k.')
               ax1.errorbar(x, y, yerr=yerr, fmt='k.')
               ax2.errorbar(np.log(x + 30), y, yerr=yerr, fmt='k.')
               
               templatePkl = "outputs/UberTemplate_%s.pkl"%(b + 'p' if b in ['u', 'r', 'i']
                                                    else b)
               tmpl = pkl.load(open(templatePkl, "rb"))
               tmpl['mu'] = -tmpl['mu']
               print ("Template for the current band", templatePkl)

               t = np.linspace(x.min(), x.max(), 100)

               tmpl['musmooth'] = -tmpl['spl'](tmpl['phs'])
               meansmooth = lambda x : -tmpl['spl'](x) + tmpl['spl'](0)  
               
               ax00.errorbar(x, y, yerr=yerr, label="data")
               ax00.plot([-25,95], [0,0], 'k-', alpha=0.5, lw=2)
               ax00.plot(x, y - meansmooth(x), label="residuals")
               ax00.plot(x, meansmooth(x), 'ko')
               ax00.plot(t, meansmooth(t), label="mean")
               ax00.legend(fontsize=20)
               ax00.set_ylim(-3,2)
               ax00.set_xlim(-30,100)
               ax00.set_title("residuals")
               ax00.set_ylabel("normalized mag")
               ax00.set_xlabel("time (days since peak)")
               
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
               if t[0]>-15:
                    #adding a point at -15
                    t=np.concatenate([np.array([-15]),t])
               mu, cov = gp.predict(y - meansmooth(x), np.log(t+30))
               std = np.sqrt(np.diag(cov))
               
               p0 = gp.kernel.vector
               
               ax01.set_title("pars: %.2f %.2f"%(p0[0], p0[1]))
               
               ax01.plot(t , mu + meansmooth(t), 'DarkOrange', lw=2)
               ax01.fill_between(t, mu - std + meansmooth(t),
                                 mu + std + meansmooth(t), color='grey', alpha=0.3)
               ax01.set_title("pars: %.2f %.2f"%(p0[0], p0[1]))
               
               # # Subracting the mean and fitting GP to residuals only
               
               #spl = InterpolatedUnivariateSpline(templ.phs, ysmooth)
               pl.savefig("outputs/GPfit%s_%s_medians.png"%(sn,b))
               #continue
               if FITGP:
                    try:
                         results = op.minimize(nll, p0,
                                               args=(y - meansmooth(x), 
                                                     x, np.log(t+30), gp, yerr))
                         #    # Update the kernel and print the final log-likelihood.
                         gp.kernel[:] = results.x
                    except :#RuntimeError:
                         #pl.savefig("GPfit%s_%s.png"%(sn,b))
                         continue
                    print ("hyper parameters: ", gp.kernel)
                    print("loglikelihood", gp.lnlikelihood(y))
               
               gp.compute(np.log(x+30), yerr )
               mu, cov = gp.predict(y - meansmooth(x), np.log(t+30))
               std = np.sqrt(np.diag(cov))
               
               p1 = gp.kernel.vector
               
               if FITGP:
                    ax1.set_xlabel("time (days since peak)")               
                    ax1.plot(t, mu + meansmooth(t), 'r', lw=2,
                             label="%.2f %.2f %s"%(p1[0], p1[1], results.success))
                    
                    ax1.fill_between(t,
                                     mu + meansmooth(t) - std, 
                                     mu + meansmooth(t) + std ,
                                     color='grey', alpha=0.3)
               
                    ax1.set_ylabel("normalized mag")
                    ax1.legend()
                    ax1.set_ylim(ax01.get_ylim())
                    ax1.set_xlim(ax01.get_xlim())        
                    ax2.set_xlabel("log time")
                    
                    ax2.plot(np.log(t+30), mu + meansmooth(t), 'r', lw=2)
                    ax2.fill_between(np.log(t+30), 
                                     mu + meansmooth(t) - std, 
                                     mu + meansmooth(t) + std ,
                                     color='grey', alpha=0.3)
                    
                    # # Subracting the mean and fitting GP to residuals only
               
                    #spl = InterpolatedUnivariateSpline(templ.phs, ysmooth)
                    
                    
                    xl = pl.xlabel("log time (starting 30 days before peak)")
                    pl.savefig("outputs/GPfit%s_%s.png"%(sn,b))
               else:
                    pkl.dump((y, gp, tmpl['spl']), open("outputs/GPfit%s_%s.pkl"%(sn,b), "wb"))
                    



