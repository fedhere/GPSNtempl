"""
This code can be used in the terminal to download light curves from the Open Supernovae Catalog.
In the terminal, go to the directory where this file is and type:
python readOSNC.py snname

If you have the time of the maximum you can give it as an argument like vmax=42929.93
Time of max should be in units of days and in JD or MJD

If you do not want to download photometry from a source existing on OSNC, first find the reference numbers of that
paper and add an argument like "2-3-4" where reference numbers are separated by dash.


"""


from __future__ import print_function, division
import os
import sys
import glob
import json
import pylab as pl
import numpy as np
import pandas as pd
cmd_folder = os.path.realpath(os.getenv("SESNCFAlib"))

if cmd_folder not in sys.path:
     sys.path.insert(0, cmd_folder)

import snclasses as snstuff

try:
     s = json.load( open(os.getenv ('PUI2018') + "/fbb_matplotlibrc.json") )
     pl.rcParams.update(s)
except (TypeError, IOError):
     pass

Nmin = 6 #minimum number of datapoints to accept a lightcurve

def doit(sn=None, url=None, vmax=None, selection_criteria = True, verbose=False, ref_remove=[]):
    # If you have already selected a set of SNe and you only need to download their 
    #lightcurves and check the criteria for the lightcurve, you should set selection_criteria = False

    #sn = 'SN2008bo'

    if url is None:
         # work locally
         print (sn)
         url = 'https://api.astrocats.space/'+str(sn)+'/sources+photometry'
         # url = "../sne.space_downloads/downloads.old/" + sn + ".json"

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
    # print('sources: ', js['sources'])
    for ref in js['sources']:
        if 'reference' in ref.keys() and 'Bianco' in ref['reference']:
            myref = ref['alias']
        if 'reference' in ref.keys() and 'Drout et al. (2011)' in ref['reference']:
             D11ref = ref['alias']

    # quit if there are no photometric datapoints
    if not 'photometry' in js.keys():
         #sys.exit()
         return
    # quit if there are fewer than N photometric datapoints
    if selection_criteria:    
        N = len(js['photometry'])
        print ("number of photometric datapoints: ", N)
        if N < Nmin:
             print ("dropping this lcvs cause it has fewer than %d datapoints"%Nmin)
             #sys.exit()
             return
    else:
        pass
    
    dtypes={'names':('mjd','w2','dw2','m2','dm2','w1','dw1','U','dU','V','dV','B','dB','R','dR','I','dI','u','du','b','db','v','dv','g','dg','r','dr','i','di','z','dz','Y','dY','J','dJ','H','dH','K','dK'),
     'formats':('f4','f4','f4','f4', 'f4','f4', 'f4','f4','f4','f4','f4','f4','f4','f4','f4','f4','f4','f4', 'f4','f4','f4','f4','f4','f4','f4','f4','f4','f4','f4', 'f4','f4','f4','f4','f4','f4','f4','f4','f4','f4')}


    snarray = np.zeros(N, dtype=dtypes)
    for i,dp in enumerate(js['photometry']):
        for j in range(len(snarray[i])):
            snarray[i][j] = np.nan 

        if 'time' in dp.keys() and 'band' in dp.keys() \
           and  dp.keys() and 'magnitude' in dp.keys():
            if verbose:
                 print ("here", dp['source'], myref, D11ref)
            # skip if the photometry is from D11 or is among the requested to-be-removed list
            if dp['source'] == D11ref or dp['source'] in ref_remove:
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
            elif band == 'UVW1': band = 'w1'
            elif band == 'W2': band = 'w2'
            elif band == 'UVW2': band = 'w2'
            elif band == 'M2': band = 'm2'

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
                 print('dmag was set to 0.01')
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
    thissn.plotsn(photometry=False, verbose=verbose)
    print(thissn.su.bands)
    thissn.formatlitsn(snarray)


if __name__ == '__main__':
    # The first argument can be the name of a single sn that you want to download from OSC
    # To input the time of Vmax if available, the argument should be vmax=...
    # You can specify what reference numbers should be ignored while reading in the light curve. You need an argument
    # with reference numbers separated by -
     if len(sys.argv) > 1:
          sn = sys.argv[1]
          print ("number of arguments", len(sys.argv))
          if len(sys.argv)>2 :
               #second argument is the max V magnitude for renormalization
               for arg in sys.argv[2:]:
                if arg.startswith('vmax='):
                    doit(sn=sn, vmax=np.float(arg.split('=')[1]))

                else:
                    ref_alias = arg.split('-')
                    doit(sn=sn, ref_remove=ref_alias)
          else:
               print ("running as doit(sn=%s, vmax=None)"%sn)
               doit(sn=sn, vmax=None, verbose=False)
               sys.exit()
     else:
          
          fbad = open("fbad.dat", "w")
          # read in the list of SNe from Khakpash+ 2022 paper from the open SN catalog:
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

               
