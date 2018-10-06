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


#s = json.load( open(os.getenv ('PUI2015')+"/fbb_matplotlibrc.json") )
#pl.rcParams.update(s)

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

avoid=["03dh"]

pl.rcParams['figure.figsize']=(10,10)


# # Loading CfA SN lightcurves

#setting parameters for lcv reader
#use literature data (if False only CfA data)
LIT = True
#use NIR data too
FNIR = True

SNTYPE = 'Ic-bl'

#pl.ion()
readgood = pd.read_csv("goodGPs.csv", header=None)
#print readgood
#sys.exit()

DEBUG = False

tcorlims = {
     'R':{'tmin':10, 'tmax':20},
     'V':{'tmin':10, 'tmax':20},     
     'r':{'tmin':10, 'tmax':20},                    
     'U':{'tmin':20, 'tmax':0},
     'u':{'tmin':20, 'tmax':0},                     
     'J':{'tmin':20, 'tmax':10},
     'B':{'tmin':20, 'tmax':5},
     'H':{'tmin':15, 'tmax':20},
     'I':{'tmin':10, 'tmax':20},
     'i':{'tmin':10, 'tmax':20},                    
     'K':{'tmin':20, 'tmax':20},
     'm2':{'tmin':20, 'tmax':20},
     'w1':{'tmin':20, 'tmax':20},
     'w2':{'tmin':20, 'tmax':20}}                    
                    

if __name__ == '__main__':

     #uncomment for all lcvs to be read in
     allsne = pd.read_csv(os.getenv("SESNCFAlib") +
                          "/SESNessentials.csv")['SNname'].values
     #set a plotcolor for each SN by assigning each to a number 0-1
     snecolors = {}

     for i,sn in enumerate(allsne):
          snecolors[sn] = i * 1.0 / (len(allsne) - 1)

     if len(sys.argv) > 1:
          if sys.argv[1] in ['Ib','IIb','Ic','Ic-bl']:
               SNTYPE = sys.argv[1]
          else:
               allsne = [sys.argv[1]]
     
     #set up SESNCfalib stuff
     su = templutils.setupvars()
     if len(sys.argv) > 2:
          bands = [sys.argv[2]]
     else:
          bands = su.bands
     nbands = len(bands)

     #errorbarInflate = {"93J":30, 
     #                   "05mf":1}

     ax = {}
     axv1 = {}
     axv2 = {}          
     figs = []
     figsv1 = []
     figsv2 = []
     
     for j,b in enumerate(bands):
          f, ax[b] = pl.subplots(2, sharex=True, sharey=True)#figsize=(20,20))
          f.subplots_adjust(hspace=0)
          figs.append(f)

          f1, axv1[b] = pl.subplots(3, sharex=True, sharey=True)#figsize=(20,20))
          f1.subplots_adjust(hspace=0)
          figsv1.append(f1)

          fv2, axv2[b] = pl.subplots(2, sharex=True, sharey=True)#figsize=(20,20))
          fv2.subplots_adjust(hspace=0)
          figsv2.append(fv2)
          
     #pl.ion()
     dt = 0.5
     t = np.arange(-15,50,dt)

     #set up arrays to host mean, mean shifted by peak, standard dev, and dtandard dev shifted
     mus = np.zeros((len(allsne), len(bands), len(t))) * np.nan
     musShifted = np.zeros((len(allsne), len(bands), len(t))) * np.nan     
     stds = np.zeros((len(allsne), len(bands), len(t))) * np.nan
     stdsShifted = np.zeros((len(allsne), len(bands), len(t))) * np.nan          
     
     for i, sn in enumerate(allsne):
          # read and set up SN and look for photometry files
          try:
               thissn = snstuff.mysn(sn, addlit=True)
          except AttributeError:
               continue
          if len(thissn.optfiles) + len(thissn.fnir) == 0:
               print ("bad sn")
          # read metadata for SN
          thissn.readinfofileall(verbose=False, earliest=False, loose=True)
          #thissn.printsn()
          if not thissn.sntype == SNTYPE:
               continue
          if thissn.snnameshort in avoid:
               continue
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

          # see if SN is in the list of good SNe with ok GP fits
          goodbad = readgood[readgood[0] == thissn.name]

          if len(goodbad)==0:
               if DEBUG:
                    raw_input()
               continue
     
          #check that it is k
          if np.array([n for n in thissn.filters.itervalues()]).sum() == 0:
               print ("bad sn")
               continue
     
          for j,b in enumerate(bands):

               tcore = (t>-tcorlims[b]['tmin']) * (t<tcorlims[b]['tmax'])
                
               if DEBUG:
                    print (goodbad)
               if len(goodbad[goodbad[1]==b]) == 0:
                    if DEBUG:
                         print ("no",b)
                         raw_input()
                    continue

               if goodbad[goodbad[1]==b][2].values[0] == 'n':
                    if DEBUG:
                         print (goodbad[goodbad[1]==b][2].values[0])
                         print ("no good",b)
                         raw_input()
                    continue
               if thissn.filters[b] == 0:
                    if DEBUG:
                         print ("no band ",b)
                         raw_input()
                    continue

               xmin = thissn.photometry[b]['mjd'].min()
               x = thissn.photometry[b]['mjd'] - thissn.Vmax + 2400000.5
               y = thissn.photometry[b]['mag'] 
               photmin = y.min()
               
               y = photmin - y
               
               yerr = thissn.photometry[b]['dmag']
        
               pklf = "outputs/GPfit%s_%s.pkl"%(sn,b)
               if not os.path.isfile(pklf):
                    print ("missing file ", pklf)
                    raw_input()
                    continue
               
               ygp, gp, tmplm = pkl.load(open(pklf, "rb"))
               meansmooth = lambda x : np.zeros_like(x)#-tmplm(x) + tmplm(0)  
               try:
                    mu, cov = gp.predict(y + meansmooth(x), np.log(t+30))
               except ValueError:
                    if DEBUG:
                         print ("error")
                         raw_input()
                    continue
               std = np.sqrt(np.diag(cov))
               if (np.abs(mu)<0.1).all():
                    continue
               mus[i][j] = mu - mu[np.abs(t)==np.abs(t).min()]
               
               stds[i][j] = std
               ax[b][0].plot(t, mus[i][j] + meansmooth(t),  lw=2,
                             label=thissn.snnameshort, alpha=0.5,
                         color = pl.cm.nipy_spectral(snecolors[sn]))

               ax[b][0].fill_between(t, 
                                  mus[i][j] + meansmooth(t) - std, 
                                  mus[i][j] + meansmooth(t) + std ,
                                  color='grey', alpha=0.1)

               axv1[b][0].plot(t, mus[i][j] + meansmooth(t),  lw=2,
                             label=thissn.snnameshort, alpha=0.5,
                         color = pl.cm.nipy_spectral(snecolors[sn]))

               axv1[b][0].fill_between(t, 
                                  mus[i][j] + meansmooth(t) - std, 
                                  mus[i][j] + meansmooth(t) + std ,
                                  color='grey', alpha=0.1)
               
               truemax = np.where(np.array(mus[i][j] + meansmooth(t)) ==
                                  (np.array(mus[i][j] + meansmooth(t))[tcore]).max())[0][0]

               if truemax < 5:
                    minloc = np.where(np.array(mus[i][j] + meansmooth(t)) ==
                         (np.array(mus[i][j] + meansmooth(t))[tcore]).min())[0][0]
                    if minloc > 0 and minloc <len(tcore):
                         tcore = (t>-10) * (t<tcorlims[b]['tmax'])    
                         truemax = np.where(np.array(mus[i][j] + meansmooth(t)) ==
                                  (np.array(mus[i][j] +
                                            meansmooth(t))[tcore]).max())[0][0]

                    if np.abs(truemax - np.where(t == t[tcore][0])[0][0]) < 2:
                         truemax = np.where(t == 0)[0][0]
               t2 = t - t[truemax]
               t20 = np.where(t2==0)[0][0]
               yoffset = (mus[i][j] + meansmooth(t))[t20]
               axv2[b][0].plot(t2, mus[i][j] + meansmooth(t) -
                               yoffset,  lw=2,
                             label=thissn.snnameshort, alpha=0.5,
                               color = pl.cm.nipy_spectral(snecolors[sn]))
               ax[b][0].scatter(t[truemax],
                                mus[i][j][truemax] + meansmooth(t[truemax]),
                                lw=2, alpha=0.5,
                                c=pl.cm.nipy_spectral(snecolors[sn]))
               axv1[b][0].scatter(t[truemax],
                                mus[i][j][truemax] + meansmooth(t[truemax]),
                                lw=2, alpha=0.5,
                                c=pl.cm.nipy_spectral(snecolors[sn]))               

               if ((SNTYPE == 'Ic' and thissn.snnameshort == '94I')
                    or (SNTYPE == 'IIb' and (thissn.snnameshort == '93J' or
                                             thissn.snnameshort == '11dh'))
                   or (SNTYPE == 'Ic-bl' and (thissn.snnameshort == '98bw' or
                                              thissn.snnameshort == '06aj'))
                   or (SNTYPE == 'Ib' and (thissn.snnameshort == '08D' ))):
                    axv1[b][0].scatter(t[truemax],
                                mus[i][j][truemax] + meansmooth(t[truemax]),
                                lw=2, alpha=1,
                                c=pl.cm.nipy_spectral(snecolors[sn]))                                   
                    axv1[b][0].errorbar(x,
                                       y, yerr,
                                        lw=1, alpha=1, fmt='.',
                                        label=thissn.snnameshort,
                                        c=pl.cm.nipy_spectral(snecolors[sn]))
                    axv1[b][1].errorbar(x,
                                       y, yerr,
                                        lw=1, alpha=1, fmt='.',
                                        label=thissn.snnameshort,
                                        c=pl.cm.nipy_spectral(snecolors[sn]))                    
                    axv1[b][2].errorbar(x - t[truemax],
                                       y, yerr,
                                        lw=1, alpha=1, fmt='.',
                                        label=thissn.snnameshort,
                                       c=pl.cm.nipy_spectral(snecolors[sn]))               

               axv2[b][0].fill_between(t2, 
                                  mus[i][j] + meansmooth(t) - std - yoffset, 
                                  mus[i][j] + meansmooth(t) + std -yoffset,
                                  color='grey', alpha=0.1)               
               tmin, tmax = t2.min(), t2.max()
               if tmin >= t[0]: 
                    musShifted[i][j][truemax:] = (mus[i][j] + meansmooth(t))[:-truemax]
                    stds[i][j][truemax:] = std[:-truemax]
               if tmin < t[0]:
                    tstart = np.where(t2 == t[0])[0][0]
                    musShifted[i][j][:-tstart] = (mus[i][j] + meansmooth(t) -
                                                  yoffset)[tstart:]
                    stdsShifted[i][j][:-tstart] = std[tstart:]
               if DEBUG:
                    print ("all the way down")
                    raw_input()                    
               # # Subracting the mean and fitting GP to residuals only
               
                    #spl = InterpolatedUnivariateSpline(templ.phs, ysmooth)
               #pl.draw() 
                    
               xl = pl.xlabel("log time (starting 30 days before peak)")
          #raw_input()

     #for k,m in enumerate(mus):
     for j,b in enumerate(bands):
          #if not b=='B':
          #     continue
          bb = b
          if b == 'i':
               bb = 'ip'
          if b == 'u':
               bb = 'up'
          if b == 'r':
               bb = 'rp'
          
          #for k,ms in enumerate(mus):
          #     if np.isnan(ms[j,:40]).all():
          #          continue    
          #     print (ms[j,:40], stds[k,j,:40])
          ax[b][0].legend(ncol=4)
          ax[b][0].set_ylim(-6,3)                  
          ax[b][0].set_xlabel("log time")
          axv1[b][0].legend(ncol=4)
          axv1[b][0].set_ylim(-6,3)                  
          axv1[b][0].set_xlabel("log time")
          #print (mus[:,:,20:40])#,stds[:,:,20:40])                    
          axv2[b][0].legend(ncol=4)
          axv2[b][0].set_ylim(-6,3)                  
          axv2[b][0].set_xlabel("log time")
          #print (mus[:,:,20:40])#,stds[:,:,20:40])          
          mask = np.isnan(mus) + ~np.isfinite(mus) +\
                                   np.isnan(stds) + ~np.isfinite(stds) +\
                                   ~np.isfinite(1.0/stds)
          maskShifted = np.isnan(musShifted) + ~np.isfinite(musShifted) +\
                                   np.isnan(stdsShifted) + ~np.isfinite(stdsShifted) +\
                                   ~np.isfinite(1.0/stdsShifted)          #mask[50:] = True

          #nlcvs = (~mask[:,j,:]).sum(1)
          #minmaxlcvs =  (np.nanmin(nlcvs[nlcvs>1]), np.nanmax(nlcvs[nlcvs>1]))
          #print (minmaxlcvs)
                                   
          #continue
          mus = np.ma.masked_array(mus, mask)
          stds = np.ma.masked_array(stds, mask)
          
          musShifted = np.ma.masked_array(musShifted, maskShifted)
          stdsShifted = np.ma.masked_array(stdsShifted, maskShifted)

          #print (mus[:,j,27:32],stds[:,j,27:32])
          average = np.ma.average(mus[:,j,:], axis=0,
                                  weights = 1.0/stds[:,j,:]**2)

          variance = np.ma.average((mus[:,j,:]-average)**2, axis=0,
                                   weights=1.0/stds[:,j,:]**2) \
                                   * np.nansum(1.0/stds[:,j,:]**2, axis=0) \
                                   / (np.nansum(1.0/stds[:,j,:]**2, axis=0) - 1)


          averageShifted = np.ma.average(musShifted[:,j,:], axis=0,
                                  weights = 1.0/stdsShifted[:,j,:]**2)

          varianceShifted = np.ma.average((musShifted[:,j,:]-averageShifted)**2, axis=0,
                                   weights=1.0/stdsShifted[:,j,:]**2) \
                                   * np.nansum(1.0/stdsShifted[:,j,:]**2, axis=0) \
                                   / (np.nansum(1.0/stdsShifted[:,j,:]**2, axis=0) - 1)
          
          std = np.ma.std( mus[:,j,:] + meansmooth(t),   axis=0)
          stdShifted = np.ma.std( musShifted[:,j,:] + meansmooth(t),   axis=0)          
          #          print (average.shape, variance.shape, meansmooth(t).shape)
          thisfit = {'t':t,
                     'average': average,
                     'averageShifted': averageShifted,
                     'variance': variance,
                     'varianceShifted': varianceShifted,
                     'stdev': std,
                     'stdShifted': stdShifted,
                     'meansmooth': meansmooth(t)}
          

          ax[b][1].plot(t, average + meansmooth(t), 'k')

          ax[b][1].fill_between(t,
                                average - std + meansmooth(t),
                                average + std + meansmooth(t),
                                color= '#1A5276', alpha=0.3 ,
                                label="sample standard deviation")                    
          ax[b][1].fill_between(t,
                                average - variance + meansmooth(t),
                                average + variance + meansmooth(t),
                                color= '#FFC600', alpha=0.3,
                                label="weighted variance")
          ax[b][0].set_title(SNTYPE + ", " + b)
          ax[b][1].set_xlabel("phase (days)")
          ax[b][0].set_ylabel("mag")
          ax[b][1].set_ylabel("mag")
          ax[b][1].legend()

          ax[b][0].set_xlim(-25,55)
          ax[b][1].set_xlim(-25,55)


          axv1[b][1].plot(t, average + meansmooth(t), 'k')

          axv1[b][1].fill_between(t,
                                average - std + meansmooth(t),
                                average + std + meansmooth(t),
                                color= '#1A5276', alpha=0.3 ,
                                label="sample standard deviation")                    
          axv1[b][1].fill_between(t,
                                average - variance + meansmooth(t),
                                average + variance + meansmooth(t),
                                color= '#FFC600', alpha=0.3,
                                label="weighted variance")
          axv1[b][0].set_title(SNTYPE + ", " + b)
          axv1[b][1].set_xlabel("phase (days)")
          axv1[b][0].set_ylabel("mag")
          axv1[b][1].set_ylabel("mag")
          axv1[b][1].legend()

          axv1[b][0].set_xlim(-25,55)
          axv1[b][1].set_xlim(-25,55)


          



          axv2[b][1].plot(t, averageShifted, 'k')

          axv2[b][1].fill_between(t,
                                averageShifted - std,
                                averageShifted + std,
                                color= '#1A5276', alpha=0.3 ,
                                label="sample standard deviation")
          axv2[b][1].fill_between(t,
                                averageShifted - varianceShifted ,
                                averageShifted + varianceShifted ,
                                color= '#FFC600', alpha=0.3,
                                label="weighted variance")
      
          axv2[b][0].set_title(SNTYPE + ", " + b)
          axv2[b][1].set_xlabel("phase (days)")
          axv2[b][0].set_ylabel("mag")
          axv2[b][1].set_ylabel("mag")
          axv2[b][1].legend()

          axv2[b][0].set_xlim(-25,55)
          axv2[b][1].set_xlim(-25,55)          
          axv2[b][0].grid(True)
          ax[b][0].grid(True)
          axv2[b][1].grid(True)
          ax[b][1].grid(True)

          axv1[b][2].plot(t, averageShifted, 'k')

          axv1[b][2].fill_between(t,
                                averageShifted - std,
                                averageShifted + std,
                                color= '#1A5276', alpha=0.3 ,
                                label="sample standard deviation")
          axv1[b][2].fill_between(t,
                                averageShifted - varianceShifted ,
                                averageShifted + varianceShifted ,
                                color= '#FFC600', alpha=0.3,
                                label="weighted variance")
      
          axv1[b][0].set_title(SNTYPE + ", " + b)
          axv1[b][2].set_xlabel("phase (days)")
          axv1[b][0].set_ylabel("mag")
          axv1[b][1].set_ylabel("mag")          
          axv1[b][2].set_ylabel("mag")
          axv1[b][1].legend()

          axv1[b][0].set_xlim(-25,55)
          axv1[b][1].set_xlim(-25,55)
          axv1[b][2].set_xlim(-25,55)                    
          axv1[b][0].grid(True)
          axv1[b][1].grid(True)
          axv1[b][2].grid(True)                    
          pkl.dump(thisfit, open("outputs/GPalltemplfit_%s_%s_V0.pkl"%(SNTYPE,bb), "wb"))
          
          #pl.figure()

          #pl.show()
          #print (variance)
          #print (average)
          #figs[j].show()
          
          figs[j].savefig("outputs/GPalltemplfit_%s_%s_V0.png"%(SNTYPE,bb))
          figsv1[j].savefig("outputs/GPalltemplfit_%s_%s_V1.pdf"%(SNTYPE,bb))
          figsv2[j].savefig("outputs/GPalltemplfit_%s_%s_V2.png"%(SNTYPE,bb))          
          os.system("pdfcrop outputs/GPalltemplfit_%s_%s_V1.pdf /Users/fbianco/science/Dropbox/papers/SESNtemplates.working/figs/GPalltemplfit_%s_%s_V1.pdf"%(SNTYPE,bb,SNTYPE,bb))



