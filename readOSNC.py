from __future__ import print_function
import pandas as pd
import numpy as np
import json
import os
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

from scipy.interpolate import InterpolatedUnivariateSpline, UnivariateSpline,splrep, splev
from scipy import interpolate
s = json.load( open(os.getenv ('PUI2015') + "/fbb_matplotlibrc.json") )
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
import glob

def doit(sn=None, url=None, vmax=None, verbose=False):
    #'SN1993J'
    #sn = 'SN2008bo'

    if url is None:
         print (sn)
         url = "../sne.space_downloads/" + sn + ".json"

    removed11 = ["07D", "06fo", "06el", "05eo","06F", "05nb", "05kz","05az","04gt"]
    if verbose: print (url, glob.glob(url))
    js = pd.read_json(url)
    snkey = js.columns
    js = js[snkey[0]]
    if verbose:
         print (js)
    myref = -99
    D11ref = -99
    for ref in js['sources']:
        if 'reference' in ref.keys() and 'Bianco' in ref['reference']:
            myref=ref['alias']
        if 'reference' in ref.keys() and 'Drout' in ref['reference']:
             D11ref = ref['alias']

    # quit if there are no photometric datapoints
    if not 'photometry' in js.keys():
         #sys.exit()
         return
    # quit if there are fewer than 5 photometric datapoints    
    N = len(js['photometry'])
    if N<=5:
         #sys.exit()
         return
    
    dtypes={'names':('mjd','w2','dw2','m2','dm2','w1','dw1','U','dU','V','dV','B','dB','R','dR','I','dI','u','du','b','db','v','dv','g','dg','r','dr','i','di','z','dz','Y','dY','J','dJ','H','dH','K','dK'),
     'formats':('f4','f4','f4','f4', 'f4','f4', 'f4','f4','f4','f4','f4','f4','f4','f4','f4','f4','f4','f4', 'f4','f4','f4','f4','f4','f4','f4','f4','f4','f4','f4', 'f4','f4','f4','f4','f4','f4','f4','f4','f4','f4')}


    snarray = np.zeros(N, dtype=dtypes)
    print ("number of photometric datapoints: ", N)
    for i,dp in enumerate(js['photometry']):
        #print (dp)
        #print ref['time']
        for j in range(len(snarray[i])):
            snarray[i][j] = np.nan 

        if 'time' in dp.keys() and 'band' in dp.keys() and  dp.keys() and 'magnitude' in dp.keys():
            if verbose:
                 print ("here", dp['source'], myref, D11ref)
            # skip if the photometry is from CfA or from D11
            if dp['source'] == myref:
                 continue
            # skip contaminated D11 data
            #print (sn)
            #print("now", sn.replace("SN20", "") in removed11)
            if dp['source'] == D11ref and sn.replace("SN20", "") in removed11:
                 continue
            #skip upper limit
            if  'upperlimit' in dp.keys():
                 continue
            band = dp['band']
            #print band
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
            if 'e_magnitude' in dp:
                 snarray[i]['d'+band.replace("'","")] = dp['e_magnitude']
            else:
                 snarray[i]['d'+band.replace("'","")] = 0.01
            if verbose:
                 print (snarray)

    thissn = snstuff.mysn(sn, noload=True, verbose=verbose)
    thissn.readinfofileall(verbose=False, earliest=False, loose=True)
    thissn.setVmax(loose=True)
    if not vmax is None:
         thissn.Vmax = vmax
         print ("Vmax", thissn.Vmax)
    if verbose: thissn.printsn(photometry=True)
    thissn.formatlitsn(snarray)


if __name__ == '__main__':
     if len(sys.argv) > 1:
          sn = sys.argv[1]
          print (len(sys.argv))
          if len(sys.argv)>2:
               doit(sn=sn, vmax=np.float(sys.argv[2]))
          else:
               print ("doit(sn=%s, vmax=None)"%sn)
               doit(sn=sn, vmax=None, verbose=True)     
     else:
          fbad = open("badlit.dat", "w")          
          sne  = open(os.getenv("DB") +
                      "/papers/SESNtemplates/tables/osnSESN.dat").readlines()
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

               
