from __future__ import print_function, division
import os
import sys
import glob 
import inspect
import optparse
import time
import copy
import json
import pylab as pl
import numpy as np
import pickle as pkl
import pandas as pd
import scipy as sp
from scipy import optimize
from scipy import stats as spstats 
from scipy import integrate
from scipy import interpolate
from scipy.interpolate import interp1d
from scipy.interpolate import InterpolatedUnivariateSpline, UnivariateSpline,splrep, splev


# =============================================================================
# cmd_folder = os.path.realpath(os.getenv("SESNCFAlib"))
# 
# if cmd_folder not in sys.path:
#      sys.path.insert(0, cmd_folder)
# =============================================================================
# sys.path.insert(0, os.path.realpath(os.getenv("SESNPATH")+ ("SESNCfAlib")))


import SESNCfAlib.snclasses as snstuff
# import templutils as templutils
# import utils as snutils
# import fitutils as fitutils
# import myastrotools as myas

try:
     s = json.load( open(os.getenv ('PUI2018') + "/fbb_matplotlibrc.json") )
     pl.rcParams.update(s)
except (TypeError, IOError):
     pass

Nmin = 6 #minimum number of datapoints to accept a lightcurve

def doit(sn=None, url=None, vmax=None, verbose=False):
    #sn = 'SN2008bo'

    if url is None:
         # work locally
         print (sn)
         url = "../sne.space_downloads/downloads.old/" + sn + ".json"

    #removing D11 data that has inconsistent photometry for objects in CfA dataset

    removed11 = ["07D", "06fo", "06el", "05eo","06F", "05nb", "05kz","05az","04gt"]
    if verbose:
         print (url, glob.glob(url))

    js = pd.read_json(url)
    snkey = js.columns
    js = js[snkey[0]]
    if verbose:
         print (js)
    #setting up dummy variables for reference notes
    myref = -99
    D11ref = -99
    for ref in js['sources']:
        if 'reference' in ref.keys() and 'Bianco' in ref['reference']:
            myref = ref['alias']
        if 'reference' in ref.keys() and 'Drout' in ref['reference']:
             D11ref = ref['alias']

    # quit if there are no photometric datapoints
    if not 'photometry' in js.keys():
         #sys.exit()
         return
    # quit if there are fewer than N photometric datapoints    
    N = len(js['photometry'])
    print ("number of photometric datapoints: ", N)
    if N < Nmin:
         print ("dropping this lcvs cause it has fewer than %d datapoints"%Nmin)
         #sys.exit()
         return
    
    dtypes={'names':('mjd','w2','dw2','m2','dm2','w1','dw1','U','dU','V','dV','B','dB','R','dR','I','dI','u','du','b','db','v','dv','g','dg','r','dr','i','di','z','dz','Y','dY','J','dJ','H','dH','K','dK'),
     'formats':('f4','f4','f4','f4', 'f4','f4', 'f4','f4','f4','f4','f4','f4','f4','f4','f4','f4','f4','f4', 'f4','f4','f4','f4','f4','f4','f4','f4','f4','f4','f4', 'f4','f4','f4','f4','f4','f4','f4','f4','f4','f4')}


    snarray = np.zeros(N, dtype=dtypes)
    for i,dp in enumerate(js['photometry']):
        #print (dp)
        #print ref['time']
        for j in range(len(snarray[i])):
            snarray[i][j] = np.nan 

        if 'time' in dp.keys() and 'band' in dp.keys() \
           and  dp.keys() and 'magnitude' in dp.keys():
            if verbose:
                 print ("here", dp['source'], myref, D11ref)
            # skip if the photometry is from CfA or from D11
            if dp['source'] == myref:
                 continue
            # skip contaminated D11 data
            # print (sn)
            # print("now", sn.replace("SN20", "") in removed11)
            if dp['source'] == D11ref and sn.replace("SN20", "") in removed11:
                 continue
            # skip upper limit
            if  'upperlimit' in dp.keys():
                 continue
            band = dp['band']
            # fix photometric band name
            if band.endswith("'"):
                 band = band.strip("'")
            if band == 'Ks': band = 'K'
            elif band == 'W1': band = 'w1'
            elif band == 'W2': band = 'w2'
            elif band == 'M2': band = 'm2'
            #print (band, dtypes['names'])
            # skip other bands
            if not band in dtypes['names']:
                 continue
            if verbose:
                 print (i, band, dp['magnitude'])

            snarray[i]['mjd'] = dp['time']
            snarray[i][band.replace("'","")] = dp['magnitude']
            #set missing uncertainty to 1% ... should probe be more!
            if 'e_magnitude' in dp:
                 snarray[i]['d'+band.replace("'","")] = dp['e_magnitude']
            else:
                 snarray[i]['d'+band.replace("'","")] = 0.01
            if verbose:
                 print (snarray)

    # use functions in SESNCfAlib library to write out the SN
    thissn = snstuff.mysn(sn, noload=True, verbose=verbose)
    thissn.readinfofileall(verbose=False, earliest=False, loose=True)
    thissn.setVmax(loose=True)
    if not vmax is None:
         thissn.Vmax = vmax
         print ("Vmax", thissn.Vmax)
    if verbose:
         thissn.printsn(photometry=True)
    thissn.plotsn(photometry=False)
    thissn.formatlitsn(snarray)


if __name__ == '__main__':
     if len(sys.argv) > 1:
          sn = sys.argv[1]
          print ("number of arguments", len(sys.argv))
          if len(sys.argv)==4 :
               #second argument is url
               #third argument is the max V magnitude for renormalization
               doit(sn=sn, url = sys.argv[2],vmax=np.float(sys.argv[3]))
          if len(sys.argv)==3 :
               #second argument is url
               doit(sn=sn, url = sys.argv[2],vmax=None, verbose=False)
          else:
               print ("running as doit(sn=%s, vmax=None)"%sn)
               doit(sn=sn, url=None, vmax=None, verbose=False)
               sys.exit()
     else:
          
          fbad = open("badlit.dat", "w")          
          # read in the list of SNe from the open SN catalog:
          # saved selection from https://sne.space/ as csv file
          sne  = pd.read_csv(open(os.getenv("GPTBLPATH") +
                                "/tables/osnSESN.csv"))["Name"]
          #sne= ["03lw"]#[sn.strip() for sn in sne]
          # D11 modification
          '''
          sne = ["04dk",
                 "04dn",
                 "04fe",
                 "04ff",
                 "04ge",
                 "04gk",
                 "04gq",
                 "04gt",
                 "04gv",
                 "05az",
                 "05hg",
                 "05kz",
                 "05la",
                 "05mf",
                 "05nb",
                 "06F", 
                 "06ab",
                 "06ck",
                 "06dn",
                 "06el",
                 "06fo",
                 "07C", 
                 "07D"]
          '''
          #sne = ["05kf"]
          for sn in sne:
               #print(sn)
               
               dontdoit = True
               try:
                    '''
                    for i in range(0,10):
                         if "SN200%s"%i in sn:
                              print ("dont skip", sn)
                              dontdoit = False
                              continue
                         

                    for i in range(10,11):
                         if "SN20%s"%i in sn:
                              print ("dont skip", sn)
                              dontdoit = False #~dontdoit                              
                              continue
                    
               
                    #for D11
                    doit(sn="SN20"+sn, vmax=None, verbose=False)
                    #if dontdoit:
                    #     continue
                    '''
                    doit(sn=sn.strip(), vmax=None, verbose=False)
               except:
                    fbad.write(sn + "\n")

               
